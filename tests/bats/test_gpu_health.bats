# shellcheck disable=SC2148
# shellcheck disable=SC2329

# Tests for GPU device health reporting (KEP-4680) and health taints
# (KEP-5055), driven by the NVML health monitor (featureGates.NVMLDeviceHealthCheck).
#
# These tests inject Xid errors through the mock NVML config override and
# therefore only run against mock NVML (MOCK_NVML=true). The mock delivers the
# configured Xid through the NVML event set after the first guarded NVML call
# in the consuming process trips the failure injector; for the kubelet plugin
# that call happens during device discovery, so the injection is followed by a
# kubelet-plugin pod restart.

setup_file() {
  load 'helpers.sh'
  _common_setup
  local _iargs=("--set" "logVerbosity=6" "--set" "featureGates.NVMLDeviceHealthCheck=true")
  if [ "${DISABLE_COMPUTE_DOMAINS:-}" = "true" ]; then
    _iargs+=("--set" "resources.computeDomains.enabled=false")
  fi
  iupgrade_wait "${TEST_CHART_REPO}" "${TEST_CHART_VERSION}" _iargs
}

teardown_file() {
  load 'helpers.sh'
  # Leave the node healthy for subsequent test files: drop the mock override
  # (if any) and restart the plugin so both the mock's tripped state and the
  # driver's sticky unhealthy state are reset.
  if [ "${MOCK_NVML:-}" = "true" ]; then
    health_clear_mock_override || true
    restart_kubelet_plugin_pods || true
  fi
}

setup() {
  load 'helpers.sh'
  _common_setup
  if [ "${MOCK_NVML:-}" != "true" ]; then
    skip "Xid injection requires mock NVML (MOCK_NVML=true)"
  fi
  log_objects
}

bats::on_failure() {
  echo -e "\n\nFAILURE HOOK START"
  log_objects
  kubectl get resourceslices -o json | jq '.items[] | select(.spec.driver=="gpu.nvidia.com") | .spec.devices[]? | {name, taints}' || true
  show_kubelet_plugin_error_logs
  show_gpu_plugin_log_tails
  echo -e "FAILURE HOOK END\n\n"
}

# --- helpers ---

# Health values reported for all DRA claims of the pod, comma-separated
# (pod.status.containerStatuses[].allocatedResourcesStatus[].resources[].health).
pod_claim_health() {
  kubectl get pod "$1" -o json \
    | jq -r '[.status.containerStatuses[]? | .allocatedResourcesStatus[]? | select(.name | startswith("claim:")) | .resources[]? | .health] | join(",")'
}

# Wait until every claim resource of the pod reports the given health.
wait_for_pod_claim_health() {
  local pod="$1" want="$2" timeout="$3"
  local start=$SECONDS got=""
  while (( SECONDS - start < timeout )); do
    got="$(pod_claim_health "${pod}")"
    if [ -n "${got}" ] && [ "$(echo "${got}" | tr ',' '\n' | sort -u)" = "${want}" ]; then
      log "pod ${pod} claim health: ${got}"
      return 0
    fi
    sleep 2
  done
  echo "Timeout (${timeout} s) waiting for claim health '${want}' on ${pod}; last: '${got}'"
  return 1
}

# Number of GPU devices in the node's ResourceSlices carrying the given taint.
count_gpu_taints() {
  local node="$1" key="$2" value="$3" effect="$4"
  kubectl get resourceslices -o json \
    | jq -r --arg n "${node}" --arg k "${key}" --arg v "${value}" --arg e "${effect}" \
      '[.items[] | select(.spec.driver=="gpu.nvidia.com" and .spec.nodeName==$n) | .spec.devices[]? | .taints[]? | select(.key==$k and .value==$v and .effect==$e)] | length'
}

wait_for_gpu_taint() {
  local node="$1" key="$2" value="$3" effect="$4" timeout="$5"
  local start=$SECONDS n=0
  while (( SECONDS - start < timeout )); do
    n="$(count_gpu_taints "${node}" "${key}" "${value}" "${effect}")"
    if [ "${n}" -gt 0 ]; then
      log "${n} device(s) on ${node} tainted with ${key}=${value}:${effect}"
      return 0
    fi
    sleep 2
  done
  echo "Timeout (${timeout} s) waiting for taint ${key}=${value}:${effect} on ${node}"
  return 1
}

kubelet_plugin_pod_on_node() {
  kubectl get pod -n dra-driver-nvidia-gpu -l dra-driver-nvidia-gpu-component=kubelet-plugin \
    --field-selector "spec.nodeName=$1,status.phase=Running" \
    --no-headers -o custom-columns=":metadata.name" | head -n1
}

# Write (or remove) the mock NVML config override through the kubelet plugin
# container: the driver root is a host path mounted at /driver-root, and the
# mock library resolves <driver_root>/config/overrides.yaml.
health_write_mock_override() {
  local plugin_pod="$1"
  kubectl exec -n dra-driver-nvidia-gpu "${plugin_pod}" -c gpus -- sh -c \
    'printf "version: 1\nall:\n  failure:\n    mode: ecc_uncorrectable\n    xid:\n      code: 79\n" > /driver-root/config/overrides.yaml && cat /driver-root/config/overrides.yaml'
}

health_clear_mock_override() {
  local pod
  for pod in $(kubectl get pod -n dra-driver-nvidia-gpu -l dra-driver-nvidia-gpu-component=kubelet-plugin \
      --field-selector status.phase=Running --no-headers -o custom-columns=":metadata.name"); do
    kubectl exec -n dra-driver-nvidia-gpu "${pod}" -c gpus -- rm -f /driver-root/config/overrides.yaml
  done
}

restart_kubelet_plugin_pods() {
  kubectl delete pod -n dra-driver-nvidia-gpu -l dra-driver-nvidia-gpu-component=kubelet-plugin --wait=true --timeout=60s
  sleep 2
  kubectl wait --for=condition=READY pods -n dra-driver-nvidia-gpu -l dra-driver-nvidia-gpu-component=kubelet-plugin --timeout=90s
}

# --- tests ---

# The kubelet reports device health as Unknown once the driver's report is
# older than the health check timeout (30 s by default). A healthy, idle GPU
# must therefore still read Healthy well past that timeout: the driver has to
# both re-send and refresh the timestamps on each monitor heartbeat.
# bats test_tags=gpu-health
@test "GPUs: health: allocated GPU is reported Healthy and stays Healthy past the kubelet health timeout" {
  local _specpath="tests/bats/specs/gpu-simple-full.yaml"
  local _podname="pod-full-gpu"

  kubectl apply -f "${_specpath}"
  kubectl wait --for=condition=READY pods "${_podname}" --timeout=30s

  if ! wait_for_pod_claim_health "${_podname}" "Healthy" 60; then
    if [ -z "$(pod_claim_health "${_podname}")" ]; then
      local minor
      minor="$(kubectl version -o json | jq -r '.serverVersion.minor' | tr -dc '0-9')"
      if [ "${minor}" -lt 36 ]; then
        kubectl delete -f "${_specpath}"
        skip "kubelet does not populate allocatedResourcesStatus (ResourceHealthStatus gate off before k8s 1.36)"
      fi
    fi
    false
  fi

  # Well past the 30 s kubelet health check timeout plus one 15 s heartbeat.
  sleep 50
  run pod_claim_health "${_podname}"
  assert_output "Healthy"

  kubectl delete -f "${_specpath}"
  kubectl wait --for=delete pods "${_podname}" --timeout=30s
}

# A critical Xid must surface both as a NoSchedule taint on the device in the
# ResourceSlice and as Unhealthy in the pod's allocatedResourcesStatus.
# bats test_tags=gpu-health
@test "GPUs: health: critical Xid taints the device and marks the allocated GPU Unhealthy" {
  local _specpath="tests/bats/specs/gpu-simple-full.yaml"
  local _podname="pod-full-gpu"

  kubectl apply -f "${_specpath}"
  kubectl wait --for=condition=READY pods "${_podname}" --timeout=30s

  local node plugin_pod
  node="$(kubectl get pod "${_podname}" -o jsonpath='{.spec.nodeName}')"
  plugin_pod="$(kubelet_plugin_pod_on_node "${node}")"
  [ -n "${plugin_pod}" ]

  # Baseline: no Xid taints, pod healthy (if the kubelet reports health).
  run count_gpu_taints "${node}" "gpu.nvidia.com/xid" "79" "NoSchedule"
  assert_output "0"
  local reports_health=true
  wait_for_pod_claim_health "${_podname}" "Healthy" 60 || reports_health=false

  # Inject: every GPU trips into ecc_uncorrectable with Xid 79 on the first
  # guarded NVML call; restart the plugin so discovery trips it and the health
  # monitor receives the event.
  health_write_mock_override "${plugin_pod}"
  restart_kubelet_plugin_pods

  wait_for_gpu_taint "${node}" "gpu.nvidia.com/xid" "79" "NoSchedule" 120

  if [ "${reports_health}" = "true" ]; then
    wait_for_pod_claim_health "${_podname}" "Unhealthy" 120
  else
    log "kubelet does not report DRA device health; skipping pod status assertion"
  fi

  # The pod itself keeps running; only scheduling of new pods onto the device
  # is blocked by the taint.
  run kubectl get pod "${_podname}" -o jsonpath='{.status.phase}'
  assert_output "Running"

  kubectl delete -f "${_specpath}"
  kubectl wait --for=delete pods "${_podname}" --timeout=30s

  # Recover: remove the override and restart the plugin; the device must be
  # advertised without the Xid taint again.
  health_clear_mock_override
  restart_kubelet_plugin_pods
  wait_for_all_gpu_resource_slices 60
  local start=$SECONDS
  while [ "$(count_gpu_taints "${node}" "gpu.nvidia.com/xid" "79" "NoSchedule")" != "0" ]; do
    (( SECONDS - start < 60 )) || { echo "Xid taint still present after recovery"; false; }
    sleep 2
  done
}
