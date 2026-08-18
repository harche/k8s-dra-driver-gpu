/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"testing"

	"github.com/stretchr/testify/require"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/dynamic-resource-allocation/resourceslice"
)

func TestNewBindingConditionsConfig(t *testing.T) {
	tests := []struct {
		name                    string
		conditions              string
		failureConditions       string
		expectedConditions      []string
		expectedFailures        []string
		expectedEnabled         bool
		expectedErrorSubstrings []string
	}{
		{
			name:            "empty by default",
			expectedEnabled: false,
		},
		{
			name:               "single condition pair",
			conditions:         "power.nvidia.com/ready",
			failureConditions:  "power.nvidia.com/failed",
			expectedConditions: []string{"power.nvidia.com/ready"},
			expectedFailures:   []string{"power.nvidia.com/failed"},
			expectedEnabled:    true,
		},
		{
			name:               "whitespace and empty entries are ignored",
			conditions:         " power.nvidia.com/ready , ,power.nvidia.com/steady ",
			failureConditions:  " power.nvidia.com/failed ",
			expectedConditions: []string{"power.nvidia.com/ready", "power.nvidia.com/steady"},
			expectedFailures:   []string{"power.nvidia.com/failed"},
			expectedEnabled:    true,
		},
		{
			name:               "conditions and failure conditions",
			conditions:         "power.nvidia.com/ready",
			failureConditions:  "power.nvidia.com/failed",
			expectedConditions: []string{"power.nvidia.com/ready"},
			expectedFailures:   []string{"power.nvidia.com/failed"},
			expectedEnabled:    true,
		},
		{
			name:                    "failure conditions without binding conditions",
			failureConditions:       "power.nvidia.com/failed",
			expectedErrorSubstrings: []string{"--binding-conditions is required"},
		},
		{
			// The apiserver rejects a device carrying one list without the
			// other, so this must fail at startup rather than at publish time.
			name:                    "binding conditions without failure conditions",
			conditions:              "power.nvidia.com/ready",
			expectedErrorSubstrings: []string{"--binding-failure-conditions is required"},
		},
		{
			name:                    "too many conditions",
			conditions:              "a.com/1,a.com/2,a.com/3,a.com/4,a.com/5",
			failureConditions:       "a.com/failed",
			expectedErrorSubstrings: []string{"at most 4"},
		},
		{
			name:                    "invalid condition type",
			conditions:              "not a valid condition!",
			failureConditions:       "power.nvidia.com/failed",
			expectedErrorSubstrings: []string{"invalid condition type"},
		},
		{
			name:                    "duplicate condition",
			conditions:              "power.nvidia.com/ready,power.nvidia.com/ready",
			failureConditions:       "power.nvidia.com/failed",
			expectedErrorSubstrings: []string{"duplicate condition type"},
		},
		{
			name:                    "overlapping condition and failure condition",
			conditions:              "power.nvidia.com/ready",
			failureConditions:       "power.nvidia.com/ready",
			expectedErrorSubstrings: []string{"listed as both"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c, err := newBindingConditionsConfig(test.conditions, test.failureConditions, powerReadinessConfig{})
			if len(test.expectedErrorSubstrings) > 0 {
				require.Error(t, err)
				for _, substr := range test.expectedErrorSubstrings {
					require.ErrorContains(t, err, substr)
				}
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.expectedConditions, c.conditions)
			require.Equal(t, test.expectedFailures, c.failureConditions)
			require.Equal(t, test.expectedEnabled, c.enabled())
		})
	}
}

// TestBindingConditionsMaxSizeMatchesAPI guards against the upstream limits
// changing underneath the validation above.
func TestBindingConditionsMaxSizeMatchesAPI(t *testing.T) {
	require.Equal(t, 4, resourceapi.BindingConditionsMaxSize)
	require.Equal(t, 4, resourceapi.BindingFailureConditionsMaxSize)
}

func TestApplyToResources(t *testing.T) {
	// Two pools, one with two slices, to cover every device the driver can
	// publish regardless of which ResourceSlice layout produced it.
	newResources := func() resourceslice.DriverResources {
		return resourceslice.DriverResources{
			Pools: map[string]resourceslice.Pool{
				"node-a": {Slices: []resourceslice.Slice{
					{Devices: []resourceapi.Device{{Name: "gpu-0"}, {Name: "gpu-1"}}},
					{Devices: []resourceapi.Device{{Name: "gpu-1-mig-1g24gb-0"}}},
				}},
				"node-b": {Slices: []resourceslice.Slice{
					{Devices: []resourceapi.Device{{Name: "gpu-0"}}},
				}},
			},
		}
	}

	t.Run("no-op when not configured", func(t *testing.T) {
		resources := newResources()
		bindingConditionsConfig{}.applyToResources(resources)
		forEachDevice(resources, func(d *resourceapi.Device) {
			require.Nil(t, d.BindingConditions, "device %s", d.Name)
			require.Nil(t, d.BindingFailureConditions, "device %s", d.Name)
		})
	})

	t.Run("stamps every device in every pool and slice", func(t *testing.T) {
		c, err := newBindingConditionsConfig("power.nvidia.com/ready", "power.nvidia.com/failed", powerReadinessConfig{})
		require.NoError(t, err)

		resources := newResources()
		c.applyToResources(resources)

		count := 0
		forEachDevice(resources, func(d *resourceapi.Device) {
			count++
			require.Equal(t, []string{"power.nvidia.com/ready"}, d.BindingConditions, "device %s", d.Name)
			require.Equal(t, []string{"power.nvidia.com/failed"}, d.BindingFailureConditions, "device %s", d.Name)
		})
		require.Equal(t, 4, count, "every published device should be stamped")
	})

	t.Run("devices do not share slices with each other or with the config", func(t *testing.T) {
		c, err := newBindingConditionsConfig("power.nvidia.com/ready", "power.nvidia.com/failed", powerReadinessConfig{})
		require.NoError(t, err)

		resources := newResources()
		c.applyToResources(resources)

		// Mutating one device's conditions must not affect any other device,
		// nor the configuration itself.
		resources.Pools["node-a"].Slices[0].Devices[0].BindingConditions[0] = "mutated"

		require.Equal(t, "power.nvidia.com/ready", c.conditions[0])
		require.Equal(t, []string{"power.nvidia.com/ready"},
			resources.Pools["node-a"].Slices[0].Devices[1].BindingConditions)
		require.Equal(t, []string{"power.nvidia.com/ready"},
			resources.Pools["node-b"].Slices[0].Devices[0].BindingConditions)
	})

	t.Run("multiple condition types are all stamped", func(t *testing.T) {
		c, err := newBindingConditionsConfig(
			"power.nvidia.com/ready,power.nvidia.com/steady", "power.nvidia.com/failed", powerReadinessConfig{})
		require.NoError(t, err)

		resources := newResources()
		c.applyToResources(resources)

		forEachDevice(resources, func(d *resourceapi.Device) {
			require.Equal(t, []string{"power.nvidia.com/ready", "power.nvidia.com/steady"},
				d.BindingConditions, "device %s", d.Name)
			require.Equal(t, []string{"power.nvidia.com/failed"},
				d.BindingFailureConditions, "device %s", d.Name)
		})
	})
}

func forEachDevice(resources resourceslice.DriverResources, fn func(*resourceapi.Device)) {
	for _, pool := range resources.Pools {
		for i := range pool.Slices {
			for j := range pool.Slices[i].Devices {
				fn(&pool.Slices[i].Devices[j])
			}
		}
	}
}

// Enabling power readiness makes the driver publish the conditions it satisfies
// itself, so an operator never has to name them.
func TestBindingConditionsWithPowerReadiness(t *testing.T) {
	t.Run("adds the driver-owned pair when nothing else is configured", func(t *testing.T) {
		c, err := newBindingConditionsConfig("", "", powerReadinessConfig{profileIDs: []int{4}})
		require.NoError(t, err)
		require.Equal(t, []string{powerReadyConditionType}, c.conditions)
		require.Equal(t, []string{powerFailedConditionType}, c.failureConditions)
		require.True(t, c.enabled())
	})

	t.Run("appends alongside operator-configured conditions", func(t *testing.T) {
		c, err := newBindingConditionsConfig("dps.example.com/ready", "dps.example.com/failed",
			powerReadinessConfig{profileIDs: []int{4}})
		require.NoError(t, err)
		require.Equal(t, []string{"dps.example.com/ready", powerReadyConditionType}, c.conditions)
		require.Equal(t, []string{"dps.example.com/failed", powerFailedConditionType}, c.failureConditions)
	})

	t.Run("does not duplicate a condition the operator already listed", func(t *testing.T) {
		c, err := newBindingConditionsConfig(powerReadyConditionType, powerFailedConditionType,
			powerReadinessConfig{profileIDs: []int{4}})
		require.NoError(t, err)
		require.Equal(t, []string{powerReadyConditionType}, c.conditions)
		require.Equal(t, []string{powerFailedConditionType}, c.failureConditions)
	})

	t.Run("adds nothing when power readiness is off", func(t *testing.T) {
		c, err := newBindingConditionsConfig("", "", powerReadinessConfig{})
		require.NoError(t, err)
		require.False(t, c.enabled())
	})

	t.Run("still respects the four-condition ceiling", func(t *testing.T) {
		_, err := newBindingConditionsConfig("a.com/1,a.com/2,a.com/3,a.com/4", "a.com/f",
			powerReadinessConfig{profileIDs: []int{4}})
		require.ErrorContains(t, err, "at most 4")
	})
}
