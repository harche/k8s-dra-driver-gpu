# shellcheck disable=SC2148
# shellcheck disable=SC2329

# Tests for KEP-5007 device binding conditions published on GPU devices.
# Requires feature gate: GPUBindingConditions=true, and a cluster with the
# DRADeviceBindingConditions and DRAResourceClaimDeviceStatus feature gates
# (both default-on since Kubernetes 1.36 and 1.37 respectively).
#
# The driver only publishes the condition types on its devices; satisfying them
# is the job of an external controller (for example an out-of-band power
# management service). These tests play that controller with kubectl, which is
# also what makes them hardware-independent: nothing here depends on what the
# external system actually does, only on the contract with the scheduler.

_BC_READY="power.nvidia.com/ready"
_BC_FAILED="power.nvidia.com/failed"

setup_file() {
  load 'helpers.sh'
  _common_setup
  local _iargs=("--set" "logVerbosity=6"
    "--set" "featureGates.GPUBindingConditions=true"
    "--set" "bindingConditions={power.nvidia.com/ready}"
    "--set" "bindingFailureConditions={power.nvidia.com/failed}")
  if [ "${DISABLE_COMPUTE_DOMAINS:-}" = "true" ]; then
    _iargs+=("--set" "resources.computeDomains.enabled=false")
  fi
  iupgrade_wait "${TEST_CHART_REPO}" "${TEST_CHART_VERSION}" _iargs
}

setup() {
  load 'helpers.sh'
  _common_setup
  log_objects
}

bats::on_failure() {
  echo -e "\n\nFAILURE HOOK START"
  log_objects
  kubectl get resourceclaims -o yaml || true
  show_kubelet_plugin_error_logs
  show_gpu_plugin_log_tails
  echo -e "FAILURE HOOK END\n\n"
}

# KEP-5007 is only usable when the cluster serves the fields: the apiserver
# silently drops bindingConditions when DRADeviceBindingConditions is disabled
# (default-off before Kubernetes 1.36). Detect that from the published slice
# rather than from a version number, so that an explicitly disabled gate is
# handled too.
_skip_unless_binding_conditions_supported() {
  local _deadline=$((SECONDS + 90)) _published=""

  # Deliberately not wait_for_all_gpu_resource_slices: that waits for *every*
  # node to publish, which does not hold on clusters where only some nodes have
  # GPUs. One slice carrying the conditions is all these tests need.
  while (( SECONDS < _deadline )); do
    _published=$(kubectl get resourceslices \
      -o jsonpath='{.items[*].spec.devices[*].bindingConditions}' 2>/dev/null)
    if [[ "${_published}" == *"${_BC_READY}"* ]]; then
      return 0
    fi
    sleep 2
  done

  # Nothing showed up. Distinguish "the driver published nothing" (a real
  # failure) from "the apiserver dropped the field" (nothing to test here).
  local _drivers
  _drivers=$(kubectl get resourceslices \
    -o custom-columns=DRIVER:.spec.driver --no-headers 2>/dev/null)
  if [[ "${_drivers}" != *"gpu.nvidia.com"* ]]; then
    echo "no gpu.nvidia.com ResourceSlices were published within 90s"
    return 1
  fi
  skip "cluster dropped bindingConditions from the published ResourceSlice; \
the DRADeviceBindingConditions feature gate is disabled"
}

# Name of the ResourceClaim generated for a pod from its ResourceClaimTemplate.
_claim_for_pod() {
  kubectl get resourceclaim \
    -o jsonpath="{.items[?(@.metadata.ownerReferences[0].name=='$1')].metadata.name}"
}

# Write a device condition into a ResourceClaim's status. This stands in for the
# external controller that KEP-5007 expects to satisfy binding conditions; the
# driver itself never writes these.
_set_device_condition() {
  local _claim="$1" _type="$2" _reason="$3" _message="$4"
  local _driver _pool _device _now _jsonpath

  _jsonpath='{.status.allocation.devices.results[0]'
  _driver=$(kubectl get resourceclaim "${_claim}" -o jsonpath="${_jsonpath}.driver}")
  _pool=$(kubectl get resourceclaim "${_claim}" -o jsonpath="${_jsonpath}.pool}")
  _device=$(kubectl get resourceclaim "${_claim}" -o jsonpath="${_jsonpath}.device}")
  _now=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  kubectl patch resourceclaim "${_claim}" --subresource=status --type=merge -p \
    "{\"status\":{\"devices\":[{\"driver\":\"${_driver}\",\"pool\":\"${_pool}\",\"device\":\"${_device}\",\"conditions\":[{\"type\":\"${_type}\",\"status\":\"True\",\"reason\":\"${_reason}\",\"message\":\"${_message}\",\"lastTransitionTime\":\"${_now}\"}]}]}}"
}

# bats test_tags=fastfeedback,gpu-binding-conditions
@test "GPUs: BindingConditions — published on GPU devices in ResourceSlices" {
  _skip_unless_binding_conditions_supported

  run kubectl get resourceslices -o jsonpath='{.items[*].spec.devices[*].bindingConditions}'
  assert_output --partial "${_BC_READY}"

  run kubectl get resourceslices -o jsonpath='{.items[*].spec.devices[*].bindingFailureConditions}'
  assert_output --partial "${_BC_FAILED}"
}

# bats test_tags=fastfeedback,gpu-binding-conditions
@test "GPUs: BindingConditions — pod stays unbound until an external controller satisfies it" {
  _skip_unless_binding_conditions_supported

  local _specpath="tests/bats/specs/gpu-binding-conditions.yaml"

  kubectl apply -f "${_specpath}"

  # The scheduler holds the pod in PreBind and says so on the pod.
  wait_for_pod_event "pod/bc-pod-0" "BindingConditionsPending" 90

  # Held before binding: the claim is allocated, but the pod has no node yet.
  run kubectl get pod bc-pod-0 -o jsonpath='{.spec.nodeName}'
  assert_output ""

  local _claim
  _claim="$(_claim_for_pod bc-pod-0)"
  assert [ -n "${_claim}" ]

  # The scheduler copies the condition types from the slice into the allocation.
  run kubectl get resourceclaim "${_claim}" \
    -o jsonpath='{.status.allocation.devices.results[0].bindingConditions}'
  assert_output --partial "${_BC_READY}"

  # No writer has run yet, so there is no per-device status at all.
  run kubectl get resourceclaim "${_claim}" -o jsonpath='{.status.devices}'
  assert_output ""

  # Now play the external controller: mark the device ready.
  _set_device_condition "${_claim}" "${_BC_READY}" "PowerStateApplied" "set by the bats suite"

  kubectl wait --for=condition=READY pods bc-pod-0 --timeout=120s

  run kubectl get pod bc-pod-0 -o jsonpath='{.spec.nodeName}'
  refute_output ""

  run kubectl logs bc-pod-0 -c ctr
  assert_output --partial "UUID: GPU-"

  kubectl delete -f "${_specpath}"
  kubectl wait --for=delete pods bc-pod-0 --timeout=60s
}

# bats test_tags=fastfeedback,gpu-binding-conditions
@test "GPUs: BindingConditions — a failure condition aborts binding and frees the claim" {
  _skip_unless_binding_conditions_supported

  local _specpath="tests/bats/specs/gpu-binding-conditions-failure.yaml"

  kubectl apply -f "${_specpath}"
  wait_for_pod_event "pod/bc-pod-1" "BindingConditionsPending" 90

  local _claim
  _claim="$(_claim_for_pod bc-pod-1)"
  assert [ -n "${_claim}" ]

  # Play the external controller again, but report failure this time.
  _set_device_condition "${_claim}" "${_BC_FAILED}" "SimulatedFailure" "set by the bats suite"

  # The scheduler surfaces the failure verbatim and then releases the device by
  # deallocating the claim, so the pod can be retried elsewhere.
  wait_for_pod_event "pod/bc-pod-1" "device binding failed" 120
  wait_for_pod_event "pod/bc-pod-1" "deallocation of ResourceClaim completed" 120

  # Still not bound: nothing ever satisfied the readiness condition.
  run kubectl get pod bc-pod-1 -o jsonpath='{.spec.nodeName}'
  assert_output ""

  kubectl delete -f "${_specpath}"
  kubectl wait --for=delete pods bc-pod-1 --timeout=60s
}
