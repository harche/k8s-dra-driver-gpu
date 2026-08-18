/*
Copyright The Kubernetes Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/dynamic-resource-allocation/resourceslice"
)

// BindingConditions (KEP-5007) let an external controller gate pod binding on
// device readiness that this driver cannot observe itself: the scheduler holds
// the pod in PreBind until every configured condition type is True in
// claim.status.devices[].conditions, and reschedules the pod if one of the
// failure condition types becomes True.
//
// This driver only publishes the condition types on its devices. Satisfying
// them is the job of whatever component performs the out-of-band work (for
// example a power-management service bringing GPUs to their target power state
// before a workload is allowed to start). That split is deliberate: the
// readiness signal lives in the external system, and KEP-5007 is designed
// around an external controller writing the conditions.
//
// Consequently, configuring condition types here without deploying a writer
// for them means every pod requesting a GPU from this driver waits for the
// scheduler's binding timeout and is then rescheduled, indefinitely. The flags
// are therefore opt-in, empty by default, and additionally gated behind the
// GPUBindingConditions feature gate.
type bindingConditionsConfig struct {
	// conditions must all be True before the scheduler binds a pod.
	conditions []string
	// failureConditions abort binding and trigger rescheduling if any is True.
	failureConditions []string
}

// enabled reports whether any binding conditions are configured. Failure
// conditions alone have no effect: the scheduler only evaluates them for
// devices that also carry at least one binding condition.
func (c bindingConditionsConfig) enabled() bool {
	return len(c.conditions) > 0
}

// parseBindingConditions parses a comma-separated list of condition types,
// ignoring surrounding whitespace and empty entries.
func parseBindingConditions(s string) []string {
	var out []string
	for _, item := range strings.Split(s, ",") {
		if item = strings.TrimSpace(item); item != "" {
			out = append(out, item)
		}
	}
	return out
}

// newBindingConditionsConfig parses and validates the raw flag values, adding
// the conditions this driver satisfies itself when power readiness is enabled.
func newBindingConditionsConfig(conditions, failureConditions string, pr powerReadinessConfig) (bindingConditionsConfig, error) {
	c := bindingConditionsConfig{
		conditions:        parseBindingConditions(conditions),
		failureConditions: parseBindingConditions(failureConditions),
	}
	if pr.enabled() {
		c.conditions = appendUnique(c.conditions, powerReadyConditionType)
		c.failureConditions = appendUnique(c.failureConditions, powerFailedConditionType)
	}
	return c, c.validate()
}

func appendUnique(list []string, value string) []string {
	if slices.Contains(list, value) {
		return list
	}
	return append(list, value)
}

// validate mirrors the API server's validation of these fields so that a
// misconfiguration fails at startup with a clear message, instead of being
// silently rejected later when the ResourceSlice is published. See
// validateDeviceBindingParameters in
// k8s.io/kubernetes/pkg/apis/resource/validation.
func (c bindingConditionsConfig) validate() error {
	var errs []error

	// The apiserver requires the two lists to be set together, and rejects any
	// ResourceSlice carrying one without the other. Catching that here matters:
	// otherwise the plugin starts, stamps every device, and then every slice
	// write is rejected, so the node's GPUs silently disappear from the cluster.
	if len(c.conditions) == 0 && len(c.failureConditions) > 0 {
		errs = append(errs, errors.New("--binding-conditions is required when --binding-failure-conditions is set: the scheduler only evaluates failure conditions for devices that also carry binding conditions"))
	}
	if len(c.failureConditions) == 0 && len(c.conditions) > 0 {
		errs = append(errs, errors.New("--binding-failure-conditions is required when --binding-conditions is set: without it a pod that cannot become ready waits for the scheduler's binding timeout instead of being rescheduled"))
	}

	for _, check := range []struct {
		flag    string
		values  []string
		maxSize int
	}{
		{"--binding-conditions", c.conditions, resourceapi.BindingConditionsMaxSize},
		{"--binding-failure-conditions", c.failureConditions, resourceapi.BindingFailureConditionsMaxSize},
	} {
		if len(check.values) > check.maxSize {
			errs = append(errs, fmt.Errorf("%s: %d condition types configured, at most %d are allowed", check.flag, len(check.values), check.maxSize))
		}
		seen := make(map[string]bool, len(check.values))
		for _, condition := range check.values {
			// Condition types are validated as label names upstream.
			for _, msg := range validation.IsQualifiedName(condition) {
				errs = append(errs, fmt.Errorf("%s: invalid condition type %q: %s", check.flag, condition, msg))
			}
			if seen[condition] {
				errs = append(errs, fmt.Errorf("%s: duplicate condition type %q", check.flag, condition))
			}
			seen[condition] = true
		}
	}

	for _, condition := range c.failureConditions {
		if slices.Contains(c.conditions, condition) {
			errs = append(errs, fmt.Errorf("condition type %q is listed as both a binding condition and a binding failure condition", condition))
		}
	}

	return errors.Join(errs...)
}

// applyToResources stamps the configured condition types onto every device in
// resources. It is a no-op when no binding conditions are configured.
//
// This is applied centrally, immediately before publishing, so that every path
// that builds ResourceSlices (partitionable devices, the flat legacy layout,
// and republishing after a device health change) is covered by construction.
func (c bindingConditionsConfig) applyToResources(resources resourceslice.DriverResources) {
	if !c.enabled() {
		return
	}
	for _, pool := range resources.Pools {
		for i := range pool.Slices {
			for j := range pool.Slices[i].Devices {
				pool.Slices[i].Devices[j].BindingConditions = slices.Clone(c.conditions)
				pool.Slices[i].Devices[j].BindingFailureConditions = slices.Clone(c.failureConditions)
			}
		}
	}
}
