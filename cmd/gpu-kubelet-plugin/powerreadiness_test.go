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
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/stretchr/testify/require"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPowerReadinessValidate(t *testing.T) {
	require.NoError(t, powerReadinessConfig{}.validate())
	require.NoError(t, powerReadinessConfig{profileIDs: []int{3, 5, 257}}.validate())
	require.ErrorContains(t, powerReadinessConfig{profileIDs: []int{2}}.validate(), "out of range",
		"2 is below the DPS range and would map to a negative bit")
	require.ErrorContains(t, powerReadinessConfig{profileIDs: []int{258}}.validate(), "out of range")
	require.ErrorContains(t, powerReadinessConfig{profileIDs: []int{5, 5}}.validate(), "duplicate profile ID")
}

func TestPowerReadinessEnabled(t *testing.T) {
	require.False(t, powerReadinessConfig{}.enabled())
	require.False(t, powerReadinessConfig{profileIDs: []int{}}.enabled())
	require.True(t, powerReadinessConfig{profileIDs: []int{3}}.enabled())
}

func TestParseProfileIDs(t *testing.T) {
	ids, err := parseProfileIDs("")
	require.NoError(t, err)
	require.Empty(t, ids)

	ids, err = parseProfileIDs(" 1 , ,3 ")
	require.NoError(t, err)
	require.Equal(t, []int{1, 3}, ids)

	_, err = parseProfileIDs("1,nope")
	require.ErrorContains(t, err, "is not a number")
}

// maskHasProfile converts a DPS profile ID to an NVML bit index before testing
// the mask. That conversion cannot be exercised on the hardware available here
// (workload power profiles are Blackwell class and later) and a mistake in it is
// silent, so it is pinned down explicitly.
func TestMaskHasProfile(t *testing.T) {
	// DPS ID 3 is NVML bit 0, per "OOB value = NVML value + 3".
	var m nvml.Mask255
	m.Mask[0] = 1
	require.True(t, maskHasProfile(m, 3), "DPS ID 3 is bit 0")
	require.False(t, maskHasProfile(m, 0), "a raw NVML index must not be accepted as a DPS ID")
	require.False(t, maskHasProfile(m, 4))

	for _, tc := range []struct {
		oobID int
		bit   int
	}{{3, 0}, {4, 1}, {34, 31}, {35, 32}, {66, 63}, {67, 64}, {257, 254}} {
		var mask nvml.Mask255
		mask.Mask[tc.bit/32] = 1 << uint(tc.bit%32)
		require.True(t, maskHasProfile(mask, tc.oobID), "DPS ID %d should be bit %d", tc.oobID, tc.bit)
		require.False(t, maskHasProfile(mask, tc.oobID+1))
	}

	// Out of range IDs never match rather than indexing outside the mask.
	full := nvml.Mask255{Mask: [8]uint32{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}}
	require.False(t, maskHasProfile(full, 2))
	require.False(t, maskHasProfile(full, 258))
	require.True(t, maskHasProfile(full, 257))
}

// writerForTest builds a writer with just enough config to exercise claim
// selection, which needs no NVML.
func writerForTest(nodeName string) *powerReadinessWriter {
	return &powerReadinessWriter{
		config: &Config{flags: &Flags{nodeName: nodeName}},
		prc:    powerReadinessConfig{profileIDs: []int{4}, interval: time.Second},
	}
}

func allocatedClaim(driver, pool, device string, conditions []string) *resourceapi.ResourceClaim {
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "claim"},
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{
				Devices: resourceapi.DeviceAllocationResult{
					Results: []resourceapi.DeviceRequestAllocationResult{{
						Driver:            driver,
						Pool:              pool,
						Device:            device,
						BindingConditions: conditions,
					}},
				},
			},
		},
	}
}

func TestDevicesToGate(t *testing.T) {
	w := writerForTest("node-a")

	t.Run("unallocated claim is ignored", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{powerReadyConditionType})
		claim.Status.Allocation = nil
		require.Empty(t, w.devicesToGate(claim))
	})

	t.Run("another driver's device is ignored", func(t *testing.T) {
		claim := allocatedClaim("other.example.com", "node-a", "gpu-0", []string{powerReadyConditionType})
		require.Empty(t, w.devicesToGate(claim))
	})

	t.Run("another node's pool is ignored", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-b", "gpu-0", []string{powerReadyConditionType})
		require.Empty(t, w.devicesToGate(claim))
	})

	t.Run("device without our condition is ignored", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{"other.example.com/ready"})
		require.Empty(t, w.devicesToGate(claim))
	})

	t.Run("device on this node with our condition is selected", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{powerReadyConditionType})
		gated := w.devicesToGate(claim)
		require.Len(t, gated, 1)
		require.Equal(t, "gpu-0", gated[0].Device)
	})

	t.Run("device already marked ready is not selected again", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{powerReadyConditionType})
		setDeviceCondition(claim, claim.Status.Allocation.Devices.Results[0], metav1.Condition{
			Type:   powerReadyConditionType,
			Status: metav1.ConditionTrue,
		})
		require.Empty(t, w.devicesToGate(claim))
	})

	t.Run("a False condition is still selected", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{powerReadyConditionType})
		setDeviceCondition(claim, claim.Status.Allocation.Devices.Results[0], metav1.Condition{
			Type:   powerReadyConditionType,
			Status: metav1.ConditionFalse,
		})
		require.Len(t, w.devicesToGate(claim), 1)
	})
}

func TestSetDeviceCondition(t *testing.T) {
	result := resourceapi.DeviceRequestAllocationResult{
		Driver: DriverName, Pool: "node-a", Device: "gpu-0",
	}
	ready := metav1.Condition{Type: powerReadyConditionType, Status: metav1.ConditionTrue, Reason: "PowerLimitSatisfied"}

	t.Run("creates an entry when the device has no status yet", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{powerReadyConditionType})
		setDeviceCondition(claim, result, ready)

		require.Len(t, claim.Status.Devices, 1)
		require.Equal(t, DriverName, claim.Status.Devices[0].Driver)
		require.Equal(t, "gpu-0", claim.Status.Devices[0].Device)
		require.Len(t, claim.Status.Devices[0].Conditions, 1)
		require.Equal(t, metav1.ConditionTrue, claim.Status.Devices[0].Conditions[0].Status)
	})

	t.Run("leaves other drivers' entries untouched", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{powerReadyConditionType})
		claim.Status.Devices = []resourceapi.AllocatedDeviceStatus{{
			Driver: "other.example.com", Pool: "node-a", Device: "nic-0",
			Conditions: []metav1.Condition{{Type: "other.example.com/ready", Status: metav1.ConditionTrue}},
		}}

		setDeviceCondition(claim, result, ready)

		require.Len(t, claim.Status.Devices, 2)
		require.Equal(t, "other.example.com", claim.Status.Devices[0].Driver)
		require.Len(t, claim.Status.Devices[0].Conditions, 1)
		require.Equal(t, DriverName, claim.Status.Devices[1].Driver)
	})

	t.Run("preserves a foreign condition on our own device entry", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{powerReadyConditionType})
		claim.Status.Devices = []resourceapi.AllocatedDeviceStatus{{
			Driver: DriverName, Pool: "node-a", Device: "gpu-0",
			Conditions: []metav1.Condition{{Type: "someone.else/thing", Status: metav1.ConditionTrue}},
		}}

		setDeviceCondition(claim, result, ready)

		require.Len(t, claim.Status.Devices, 1)
		require.Len(t, claim.Status.Devices[0].Conditions, 2)
		require.Equal(t, "someone.else/thing", claim.Status.Devices[0].Conditions[0].Type)
		require.Equal(t, powerReadyConditionType, claim.Status.Devices[0].Conditions[1].Type)
	})

	t.Run("flips an existing condition rather than duplicating it", func(t *testing.T) {
		claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{powerReadyConditionType})
		setDeviceCondition(claim, result, metav1.Condition{Type: powerReadyConditionType, Status: metav1.ConditionFalse})
		setDeviceCondition(claim, result, ready)

		require.Len(t, claim.Status.Devices, 1)
		require.Len(t, claim.Status.Devices[0].Conditions, 1)
		require.Equal(t, metav1.ConditionTrue, claim.Status.Devices[0].Conditions[0].Status)
		require.Equal(t, "PowerLimitSatisfied", claim.Status.Devices[0].Conditions[0].Reason)
	})
}

func TestConditionIsTrue(t *testing.T) {
	result := resourceapi.DeviceRequestAllocationResult{
		Driver: DriverName, Pool: "node-a", Device: "gpu-0",
	}
	claim := allocatedClaim(DriverName, "node-a", "gpu-0", []string{powerReadyConditionType})

	require.False(t, conditionIsTrue(claim, result, powerReadyConditionType))

	setDeviceCondition(claim, result, metav1.Condition{Type: powerReadyConditionType, Status: metav1.ConditionFalse})
	require.False(t, conditionIsTrue(claim, result, powerReadyConditionType))

	setDeviceCondition(claim, result, metav1.Condition{Type: powerReadyConditionType, Status: metav1.ConditionTrue})
	require.True(t, conditionIsTrue(claim, result, powerReadyConditionType))

	// A different device on the same claim must not be confused with ours.
	other := resourceapi.DeviceRequestAllocationResult{
		Driver: DriverName, Pool: "node-a", Device: "gpu-1",
	}
	require.False(t, conditionIsTrue(claim, other, powerReadyConditionType))
}

// profileMask builds an nvmlMask255_t with the given profile IDs set.
// profileMask builds a mask from DPS profile IDs, applying the same offset the
// GPU firmware reports through NVML.
func profileMask(oobIDs ...int) nvml.Mask255 {
	var m nvml.Mask255
	for _, id := range oobIDs {
		bit := id - oobProfileIDOffset
		m.Mask[bit/32] |= 1 << uint(bit%32)
	}
	return m
}

// writerWithGPU returns a writer whose node has one GPU named gpu-0, with the
// NVML read stubbed out.
func writerWithGPU(profileIDs []int, read func(uuid string) (nvml.WorkloadPowerProfileCurrentProfiles, nvml.Return)) *powerReadinessWriter {
	return &powerReadinessWriter{
		config: &Config{flags: &Flags{nodeName: "node-a"}},
		state: &DeviceState{
			perGPUAllocatable: &PerGPUAllocatableDevices{
				allocatablesMap: map[PCIBusID]AllocatableDevices{
					"0000:00:04.0": {"gpu-0": &AllocatableDevice{Gpu: &GpuInfo{UUID: "GPU-test"}}},
				},
			},
		},
		prc:                 powerReadinessConfig{profileIDs: profileIDs, interval: time.Second},
		readCurrentProfiles: read,
	}
}

func TestDeviceHasProfiles(t *testing.T) {
	enforced := func(ids ...int) func(string) (nvml.WorkloadPowerProfileCurrentProfiles, nvml.Return) {
		return func(string) (nvml.WorkloadPowerProfileCurrentProfiles, nvml.Return) {
			return nvml.WorkloadPowerProfileCurrentProfiles{EnforcedProfilesMask: profileMask(ids...)}, nvml.SUCCESS
		}
	}

	t.Run("ready when every configured profile is enforced", func(t *testing.T) {
		w := writerWithGPU([]int{4, 6}, enforced(4, 6, 10))
		ready, err := w.deviceHasProfiles("gpu-0")
		require.NoError(t, err)
		require.True(t, ready)
	})

	t.Run("not ready when one profile is missing", func(t *testing.T) {
		w := writerWithGPU([]int{4, 6}, enforced(4))
		ready, err := w.deviceHasProfiles("gpu-0")
		require.NoError(t, err)
		require.False(t, ready)
	})

	t.Run("requested but not enforced does not count", func(t *testing.T) {
		// A profile can be requested and then lose arbitration. Only the
		// enforced mask says what is actually active on the GPU.
		w := writerWithGPU([]int{4}, func(string) (nvml.WorkloadPowerProfileCurrentProfiles, nvml.Return) {
			return nvml.WorkloadPowerProfileCurrentProfiles{
				RequestedProfilesMask: profileMask(4),
				EnforcedProfilesMask:  profileMask(5),
			}, nvml.SUCCESS
		})
		ready, err := w.deviceHasProfiles("gpu-0")
		require.NoError(t, err)
		require.False(t, ready)
	})

	t.Run("hardware without profile support is treated as ready", func(t *testing.T) {
		w := writerWithGPU([]int{4}, func(string) (nvml.WorkloadPowerProfileCurrentProfiles, nvml.Return) {
			return nvml.WorkloadPowerProfileCurrentProfiles{}, nvml.ERROR_NOT_SUPPORTED
		})
		ready, err := w.deviceHasProfiles("gpu-0")
		require.NoError(t, err)
		require.True(t, ready, "blocking on a GPU that cannot be asked would strand the pod")
	})

	t.Run("other NVML errors surface", func(t *testing.T) {
		w := writerWithGPU([]int{4}, func(string) (nvml.WorkloadPowerProfileCurrentProfiles, nvml.Return) {
			return nvml.WorkloadPowerProfileCurrentProfiles{}, nvml.ERROR_GPU_IS_LOST
		})
		_, err := w.deviceHasProfiles("gpu-0")
		require.ErrorContains(t, err, "reading workload power profiles")
	})

	t.Run("a device with no resolvable GPU is treated as ready", func(t *testing.T) {
		w := writerWithGPU([]int{4}, enforced(4))
		ready, err := w.deviceHasProfiles("gpu-does-not-exist")
		require.NoError(t, err)
		require.True(t, ready)
	})
}
