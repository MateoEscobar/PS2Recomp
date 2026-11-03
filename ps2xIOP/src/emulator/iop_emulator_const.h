#pragma once
#include <cstdint>

constexpr uint32_t kThreadReturnSentinel = 0x1FFFFF00u;
constexpr uint32_t kCallReturnSentinel = 0x1FFFFF04u;
constexpr uint64_t kIopClockHz = 36'864'000ull;
// NTSC field cadence (approximately 59.94 Hz).  VBlank imports are
// scheduler waits, not no-op timing hints: returning immediately lets
// high-priority IRX threads busy-loop and starve RPC server threads.
constexpr uint64_t kVblankPeriodCycles = (kIopClockHz * 1001ull + 30'000ull) / 60'000ull;
constexpr uint64_t kVblankEndPhaseCycles = kVblankPeriodCycles / 16ull;
constexpr uint32_t kDefaultSlice = 256u;
// Safety net for synchronous guest calls (RPC handlers, module entry points, CD
// callbacks), not a time slice: a legitimate handler may run for millions of
// instructions (MultiStream's SFX registration is O(n^2) over 512 slots and needs
// > 2M), and cutting it leaves the IRX half-updated. 128M is ~3.5 s of real IOP
// time; reaching it is logged as an error because it means a real hang.
constexpr uint32_t kMaxCallInstructions = 128'000'000u;
constexpr uint32_t kModuleLoadBase = 0x00010000u;
constexpr uint32_t kStackGuardBytes = 64u;