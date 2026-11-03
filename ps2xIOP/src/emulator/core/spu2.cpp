// SPU2 emulation, ported from god-of-war-recomp (GPLv3). See spu2.h.
#include "spu2.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr int32_t kAdpcmFilters[5][2] = {{0, 0}, {60, 0}, {115, -52}, {98, -55}, {122, -60}};
        constexpr size_t kMaxBufferedFrames = 48000u * 2u; // 2 s; si nadie consume, se descarta lo mas antiguo

        int16_t clamp16(int32_t value) noexcept
        {
            return static_cast<int16_t>(std::clamp(value, -32768, 32767));
        }

        // PS2X_SPU2_PERF=1 measures, on the thread that advances the IOP, the elapsed time of sample
        // generation (advanceTo) and DMA, and on the audio thread the drain. PS2X_SPU2_PERF=2 adds TSC
        // ticks per section (voices, envelopes, ADPCM decode, output), calibrated against advanceTo. These
        // are elapsed times of one thread, not exclusive CPU use; level 2 has its own per-voice overhead.
        int perfLevel() noexcept
        {
            static const int level = []
            {
                const char *value = std::getenv("PS2X_SPU2_PERF");
                return value ? std::atoi(value) : 0;
            }();
            return level;
        }

        struct PerfCounters
        {
            std::atomic<uint64_t> samples{0}, advanceCalls{0}, stepNs{0}, activeVoiceSteps{0}, decodedBlocks{0};
            std::atomic<uint64_t> dmaNs{0}, dmaBytes{0}, dmaCalls{0};
            std::atomic<uint64_t> drainNs{0}, drainLockNs{0}, drainCalls{0}, requestedFrames{0}, deliveredFrames{0};
            std::atomic<uint64_t> latencyDroppedFrames{0}, bufferDroppedFrames{0};
            std::atomic<uint64_t> stepTicks{0}, voiceTicks{0}, envelopeTicks{0}, decodeTicks{0}, outputTicks{0};
            std::atomic<uint64_t> outputLockTicks{0};
        };
        PerfCounters g_perf;

        uint64_t perfNow() noexcept
        {
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        }

        // Ciclos TSC en x86; en otras arquitecturas, nanosegundos (se calibran igual contra advanceTo).
        uint64_t perfTicks() noexcept
        {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
            return __rdtsc();
#else
            return perfNow();
#endif
        }

        // Una línea cada 5 s, desde el hilo que avanza el IOP (el único que produce muestras).
        void perfReport(uint64_t now)
        {
            static uint64_t first = now, last = now;
            static std::array<uint64_t, 21> before{};
            if (now - last < 5000000000ull)
                return;
            const double window = static_cast<double>(now - last) * 1e-9;
            const std::array<uint64_t, 21> current = {
                g_perf.samples.load(), g_perf.advanceCalls.load(), g_perf.stepNs.load(), g_perf.activeVoiceSteps.load(),
                g_perf.decodedBlocks.load(), g_perf.dmaNs.load(), g_perf.dmaBytes.load(), g_perf.dmaCalls.load(),
                g_perf.drainNs.load(), g_perf.drainLockNs.load(), g_perf.drainCalls.load(), g_perf.requestedFrames.load(),
                g_perf.deliveredFrames.load(), g_perf.latencyDroppedFrames.load(), g_perf.bufferDroppedFrames.load(),
                g_perf.stepTicks.load(), g_perf.voiceTicks.load(), g_perf.envelopeTicks.load(), g_perf.decodeTicks.load(),
                g_perf.outputTicks.load(), g_perf.outputLockTicks.load()};
            std::array<uint64_t, 21> d{};
            for (size_t i = 0; i < d.size(); ++i)
                d[i] = current[i] - before[i];
            before = current;
            last = now;
            const double samples = static_cast<double>(d[0]);
            std::fprintf(stderr,
                         "[spu2-perf] t=%.2f ventana=%.2f muestras=%llu muestras_s=%.0f step_ms=%.2f us_muestra=%.3f "
                         "advance_con_muestras=%llu voces_activas_med=%.2f bloques_adpcm=%llu dma_ms=%.3f dma_kb=%.1f "
                         "dma_llamadas=%llu drenado_ms=%.3f drenado_espera_mutex_ms=%.3f drenado_llamadas=%llu pedidas=%llu "
                         "entregadas=%llu faltantes=%llu descartadas_latencia=%llu descartadas_buffer=%llu\n",
                         (now - first) * 1e-9, window, static_cast<unsigned long long>(d[0]), samples / window, d[2] * 1e-6,
                         samples > 0.0 ? d[2] * 1e-3 / samples : 0.0, static_cast<unsigned long long>(d[1]),
                         samples > 0.0 ? d[3] / samples : 0.0, static_cast<unsigned long long>(d[4]), d[5] * 1e-6,
                         d[6] / 1024.0, static_cast<unsigned long long>(d[7]), d[8] * 1e-6, d[9] * 1e-6,
                         static_cast<unsigned long long>(d[10]), static_cast<unsigned long long>(d[11]),
                         static_cast<unsigned long long>(d[12]),
                         static_cast<unsigned long long>(d[11] > d[12] ? d[11] - d[12] : 0u),
                         static_cast<unsigned long long>(d[13]), static_cast<unsigned long long>(d[14]));
            if (perfLevel() >= 2 && d[15] != 0u)
            {
                // Ciclos TSC a ms con la relación de esta ventana (tiempo de advanceTo / ciclos de step).
                const double msPerTick = d[2] * 1e-6 / static_cast<double>(d[15]);
                const double voices = d[16] * msPerTick, envelope = d[17] * msPerTick, decode = d[18] * msPerTick;
                std::fprintf(stderr,
                             "[spu2-perf] secciones voces_ms=%.2f envolvente_ms=%.2f decodificacion_ms=%.2f "
                             "tono_interpolacion_mezcla_ms=%.2f salida_ms=%.2f salida_espera_mutex_ms=%.3f\n",
                             voices, envelope, decode, voices - envelope - decode, d[19] * msPerTick, d[20] * msPerTick);
            }
        }

        // Voice or core volume. Fixed mode (bit 15 = 0): signed 15 bits. Sweeps (bit 15 = 1) are not emulated
        // yet: their target applies at once (maximum when rising, 0 when falling).
        int32_t volumeFromRegister(uint16_t value) noexcept
        {
            if ((value & 0x8000u) == 0u)
                return static_cast<int16_t>(value << 1u) >> 1; // -0x4000..0x3FFF
            return (value & 0x2000u) != 0u ? 0 : 0x3FFF;
        }
    }

    Spu2::Spu2() : m_ram(RamHalfwords)
    {
        reset();
    }

    void Spu2::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint16_t{0});
        m_regs.fill(0);
        for (auto &core : m_voices)
            for (auto &voice : core)
                voice = VoiceState{};
        m_endx.fill(0);
        {
            std::lock_guard<std::mutex> lock(m_outputMutex);
            m_output.clear();
        }
        m_lastCycle = 0;
        m_interrupt = false;
        reg(RegStatx) = StatxDmaReady;
        reg(0x400u + RegStatx) = StatxDmaReady;
    }

    uint32_t Spu2::reg32(uint32_t offset) const noexcept
    {
        return ((static_cast<uint32_t>(reg(offset)) & 0xFu) << 16u) | reg(offset + 2u);
    }

    void Spu2::setReg32(uint32_t offset, uint32_t value) noexcept
    {
        value &= RamHalfwords - 1u;
        reg(offset) = static_cast<uint16_t>(value >> 16u);
        reg(offset + 2u) = static_cast<uint16_t>(value);
    }

    void Spu2::writeRam(uint32_t halfwordAddress, uint16_t value) noexcept
    {
        halfwordAddress &= RamHalfwords - 1u;
        m_ram[halfwordAddress] = value;
        checkIrqAddress(halfwordAddress);
    }

    void Spu2::checkIrqAddress(uint32_t halfwordAddress) noexcept
    {
        for (uint32_t core = 0; core < 2u; ++core)
        {
            const uint32_t base = core * 0x400u;
            if ((reg(base + RegAttr) & 0x40u) != 0u && reg32(base + RegIrqa) == halfwordAddress)
            {
                m_interrupt = true;
                reg(0x7C2u) |= static_cast<uint16_t>(4u << core); // SPDIF_IRQINFO
            }
        }
    }

    bool Spu2::takeInterrupt() noexcept
    {
        const bool pending = m_interrupt;
        m_interrupt = false;
        return pending;
    }

    uint16_t Spu2::read16(uint32_t physical) const
    {
        const uint32_t offset = (physical - RegBase) & 0x7FEu;
        if (offset < 0x760u)
        {
            const uint32_t core = offset >= 0x400u ? 1u : 0u;
            const uint32_t local = offset & 0x3FFu;
            if (local < 0x180u)
            {
                const uint32_t index = local >> 4u;
                const uint32_t field = local & 0xFu;
                if (index < VoicesPerCore)
                {
                    const VoiceState &v = m_voices[core][index];
                    if (field == 0xAu)
                        return static_cast<uint16_t>(v.envelope);
                    if (field == 0xCu || field == 0xEu)
                        return static_cast<uint16_t>(volumeFromRegister(reg(voiceReg(core, index, field - 0xCu))) << 1u);
                }
            }
            else if (local >= RegVoiceAddr && local < RegVoiceAddr + VoicesPerCore * 0xCu)
            {
                const uint32_t index = (local - RegVoiceAddr) / 0xCu;
                const uint32_t field = (local - RegVoiceAddr) % 0xCu;
                // NAX as PCSX2 reports NextA (inside the playing block). Streams that poll NAX (989snd in God of War, MultiStream in The Warriors) kill
                // the stream (error 0x35) when NAX falls outside both ring halves.
                if (field == 8u)
                    return static_cast<uint16_t>(naxRegister(m_voices[core][index]) >> 16u);
                if (field == 10u)
                    return static_cast<uint16_t>(naxRegister(m_voices[core][index]));
            }
            else if (local == RegEndx)
                return static_cast<uint16_t>(m_endx[core]);
            else if (local == RegEndx + 2u)
                return static_cast<uint16_t>(m_endx[core] >> 16u);
        }
        return reg(offset);
    }

    // value read from the NAX register pair. PCSX2 (Mixer.cpp DecodeSamples/IncrementNextA) keeps NextA
    // in [block + 1, block + 7] and jumps to LoopStartA as soon as a loop-end block finishes, so NAX never points
    // past the end of a block. Here 'nax' is the next block to decode, which during the last block of a streaming
    // half equals the first address after that half; God of War's 989snd (989NOMID.IRX 0x13258) then sees NAX outside both
    // ring halves and aborts the stream (error 0x35), so the music restarts every few seconds.
    uint32_t Spu2::naxRegister(const VoiceState &v) noexcept
    {
        if (!v.playStarted)
            return (v.playBlock | 1u) & (RamHalfwords - 1u);
        const uint32_t played = std::min<uint32_t>(v.decodedIndex, 27u) / 4u; // 0..6 (4 samples per halfword)
        return (v.playBlock + 1u + played) & (RamHalfwords - 1u);
    }

    void Spu2::write16(uint32_t physical, uint16_t value)
    {
        const uint32_t offset = (physical - RegBase) & 0x7FEu;
        reg(offset) = value;
        if (offset >= 0x760u)
            return;

        const uint32_t core = offset >= 0x400u ? 1u : 0u;
        const uint32_t base = core * 0x400u;
        const uint32_t local = offset & 0x3FFu;
        switch (local)
        {
        case RegKon:
            keyOn(core, value, 0u);
            return;
        case RegKon + 2u:
            keyOn(core, value & 0xFFu, 16u);
            return;
        case RegKoff:
            keyOff(core, value, 0u);
            return;
        case RegKoff + 2u:
            keyOff(core, value & 0xFFu, 16u);
            return;
        case RegData:
        {
            uint32_t tsa = reg32(base + RegTsa);
            writeRam(tsa, value);
            setReg32(base + RegTsa, tsa + 1u);
            return;
        }
        case RegEndx:
            m_endx[core] &= 0xFF0000u;
            return;
        case RegEndx + 2u:
            m_endx[core] &= 0x00FFFFu;
            return;
        default:
            break;
        }

        if (local >= RegVoiceAddr && local < RegVoiceAddr + VoicesPerCore * 0xCu)
        {
            const uint32_t index = (local - RegVoiceAddr) / 0xCu;
            const uint32_t field = (local - RegVoiceAddr) % 0xCu;
            VoiceState &v = m_voices[core][index];
            if (field == 4u || field == 6u) // LSAX escrito por el driver: manda sobre la marca del bloque
            {
                v.lsa = reg32(voiceAddrReg(core, index, 4u));
                v.lsaFromRegister = true;
            }
            else if (field == 8u || field == 10u)
            {
                v.nax = reg32(voiceAddrReg(core, index, 8u)) & ~7u;
                v.playBlock = v.nax; // reads back as written (| 1) until the next block is decoded
                v.playStarted = false;
            }
        }
    }

    bool Spu2::dmaWrite(uint32_t core, const uint8_t *source, size_t bytes)
    {
        const uint32_t base = core * 0x400u;
        if ((reg(base + RegAdmas) & (core + 1u)) != 0u)
            return false;
        const uint64_t start = perfLevel() ? perfNow() : 0u;
        uint32_t tsa = reg32(base + RegTsa);
        for (size_t i = 0; i + 1u < bytes; i += 2u)
            writeRam(tsa++, static_cast<uint16_t>(source[i] | (source[i + 1u] << 8u)));
        setReg32(base + RegTsa, tsa);
        if (start != 0u)
        {
            g_perf.dmaNs.fetch_add(perfNow() - start, std::memory_order_relaxed);
            g_perf.dmaBytes.fetch_add(bytes, std::memory_order_relaxed);
            g_perf.dmaCalls.fetch_add(1u, std::memory_order_relaxed);
        }
        return true;
    }

    bool Spu2::dmaRead(uint32_t core, uint8_t *destination, size_t bytes)
    {
        const uint32_t base = core * 0x400u;
        if ((reg(base + RegAdmas) & (core + 1u)) != 0u)
            return false;
        uint32_t tsa = reg32(base + RegTsa);
        for (size_t i = 0; i + 1u < bytes; i += 2u)
        {
            const uint16_t value = ram(tsa++);
            destination[i] = static_cast<uint8_t>(value);
            destination[i + 1u] = static_cast<uint8_t>(value >> 8u);
        }
        setReg32(base + RegTsa, tsa);
        return true;
    }

    void Spu2::markDmaReady(uint32_t core) noexcept
    {
        reg(core * 0x400u + RegStatx) |= StatxDmaReady;
    }

    void Spu2::keyOn(uint32_t core, uint32_t mask, uint32_t firstVoice)
    {
        for (uint32_t bit = 0; bit < 16u; ++bit)
        {
            if ((mask & (1u << bit)) == 0u || firstVoice + bit >= VoicesPerCore)
                continue;
            const uint32_t index = firstVoice + bit;
            VoiceState &v = m_voices[core][index];
            const uint32_t ssa = reg32(voiceAddrReg(core, index, 0u)) & ~7u;
            v = VoiceState{};
            v.active = true;
            v.nax = ssa;
            v.lsa = ssa;
            v.playBlock = ssa; // NAX reads SSA | 1 until the first block is decoded (PCSX2 Start())
            v.phase = VoiceState::Phase::Attack;
            m_endx[core] &= ~(1u << index);
        }
    }

    void Spu2::keyOff(uint32_t core, uint32_t mask, uint32_t firstVoice)
    {
        for (uint32_t bit = 0; bit < 16u; ++bit)
        {
            if ((mask & (1u << bit)) == 0u || firstVoice + bit >= VoicesPerCore)
                continue;
            VoiceState &v = m_voices[core][firstVoice + bit];
            if (v.active)
            {
                v.phase = VoiceState::Phase::Release;
                v.envelopeCounter = 0;
            }
        }
    }

    void Spu2::decodeBlock(uint32_t core, uint32_t index)
    {
        VoiceState &v = m_voices[core][index];

        // Fin del bloque anterior: con la marca de fin se salta a LSA; sin "repetir", la voz se apaga.
        if ((v.blockFlags & 1u) != 0u)
        {
            m_endx[core] |= 1u << index;
            v.nax = v.lsa;
            if ((v.blockFlags & 2u) == 0u)
            {
                v.active = false;
                v.phase = VoiceState::Phase::Off;
                v.envelope = 0;
            }
        }
        v.blockFlags = 0;
        if (!v.active)
            return;

        if (perfLevel())
            g_perf.decodedBlocks.fetch_add(1u, std::memory_order_relaxed);
        const uint32_t block = v.nax & (RamHalfwords - 1u);
        const uint16_t header = ram(block);
        uint32_t shift = header & 0xFu;
        if (shift > 12u)
            shift = 9u;
        const uint32_t filter = std::min<uint32_t>((header >> 4u) & 7u, 4u);
        v.blockFlags = static_cast<uint8_t>(header >> 8u);
        if ((v.blockFlags & 4u) != 0u && !v.lsaFromRegister)
            v.lsa = block;

        for (uint32_t word = 1; word < 8u; ++word)
        {
            const uint16_t packed = ram(block + word);
            checkIrqAddress((block + word) & (RamHalfwords - 1u));
            for (uint32_t nibble = 0; nibble < 4u; ++nibble)
            {
                const int32_t raw = static_cast<int16_t>(((packed >> (nibble * 4u)) & 0xFu) << 12u) >> shift;
                const int32_t predicted = (v.hist1 * kAdpcmFilters[filter][0] + v.hist2 * kAdpcmFilters[filter][1] + 32) >> 6;
                const int16_t sample = clamp16(raw + predicted);
                v.hist2 = v.hist1;
                v.hist1 = sample;
                v.decoded[(word - 1u) * 4u + nibble] = sample;
            }
        }
        checkIrqAddress(block);
        v.decodedIndex = 0;
        v.nax = (block + 8u) & (RamHalfwords - 1u);
        v.playBlock = block; // see naxRegister()
        v.playStarted = true;
    }

    void Spu2::stepEnvelope(uint32_t core, uint32_t index)
    {
        VoiceState &v = m_voices[core][index];
        const uint16_t adsr1 = reg(voiceReg(core, index, 6u));
        const uint16_t adsr2 = reg(voiceReg(core, index, 8u));

        uint32_t rate = 0;
        bool exponential = false;
        bool increase = false;
        switch (v.phase)
        {
        case VoiceState::Phase::Attack:
            rate = (adsr1 >> 8u) & 0x7Fu;
            exponential = (adsr1 & 0x8000u) != 0u;
            increase = true;
            break;
        case VoiceState::Phase::Decay:
            rate = ((adsr1 >> 4u) & 0xFu) * 4u;
            exponential = true;
            break;
        case VoiceState::Phase::Sustain:
            rate = (adsr2 >> 6u) & 0x7Fu;
            exponential = (adsr2 & 0x8000u) != 0u;
            increase = (adsr2 & 0x4000u) == 0u;
            break;
        case VoiceState::Phase::Release:
            rate = (adsr2 & 0x1Fu) * 4u;
            exponential = (adsr2 & 0x20u) != 0u;
            break;
        case VoiceState::Phase::Off:
            return;
        }

        // psx-spx "SPU ADSR": ciclos = 1 << max(0, shift - 11); paso = valor << max(0, 11 - shift).
        const int32_t shiftValue = static_cast<int32_t>(rate >> 2u);
        const int32_t stepValue = static_cast<int32_t>(rate & 3u);
        int32_t step = increase ? 7 - stepValue : -8 + stepValue;
        uint32_t cycles = 1u << static_cast<uint32_t>(std::max(0, shiftValue - 11));
        step <<= std::max(0, 11 - shiftValue);
        if (exponential && increase && v.envelope > 0x6000)
            cycles *= 4u;
        if (exponential && !increase)
            step = step * v.envelope / 0x8000;

        if (++v.envelopeCounter >= cycles)
        {
            v.envelopeCounter = 0;
            v.envelope = std::clamp(v.envelope + step, 0, 0x7FFF);
        }

        switch (v.phase)
        {
        case VoiceState::Phase::Attack:
            if (v.envelope >= 0x7FFF)
            {
                v.phase = VoiceState::Phase::Decay;
                v.envelopeCounter = 0;
            }
            break;
        case VoiceState::Phase::Decay:
            if (v.envelope <= static_cast<int32_t>(((adsr1 & 0xFu) + 1u) * 0x800u))
            {
                v.phase = VoiceState::Phase::Sustain;
                v.envelopeCounter = 0;
            }
            break;
        case VoiceState::Phase::Release:
            if (v.envelope <= 0)
            {
                v.envelope = 0;
                v.phase = VoiceState::Phase::Off;
                v.active = false;
            }
            break;
        default:
            break;
        }
    }

    int32_t Spu2::stepVoice(uint32_t core, uint32_t index)
    {
        VoiceState &v = m_voices[core][index];
        if (!v.active)
            return 0;
        const int level = perfLevel();
        if (level)
            g_perf.activeVoiceSteps.fetch_add(1u, std::memory_order_relaxed);

        // Tono 4.12 (0x1000 = 48 kHz). Interpolacion lineal entre las dos ultimas muestras (el hardware usa una
        // gaussiana de 4 puntos).
        v.pitchCounter += std::min<uint32_t>(reg(voiceReg(core, index, 4u)), 0x3FFFu);
        while (v.pitchCounter >= 0x1000u && v.active)
        {
            v.pitchCounter -= 0x1000u;
            if (v.decodedIndex >= 28u)
            {
                const uint64_t decodeStart = level >= 2 ? perfTicks() : 0u;
                decodeBlock(core, index);
                if (decodeStart != 0u)
                    g_perf.decodeTicks.fetch_add(perfTicks() - decodeStart, std::memory_order_relaxed);
            }
            if (!v.active)
                break;
            v.prev[1] = v.prev[0];
            v.prev[0] = v.decoded[v.decodedIndex++];
        }
        if (!v.active)
            return 0;

        const int32_t fraction = static_cast<int32_t>(v.pitchCounter & 0xFFFu);
        const int32_t sample = v.prev[1] + (((v.prev[0] - v.prev[1]) * fraction) >> 12);
        const uint64_t envelopeStart = level >= 2 ? perfTicks() : 0u;
        stepEnvelope(core, index);
        if (envelopeStart != 0u)
            g_perf.envelopeTicks.fetch_add(perfTicks() - envelopeStart, std::memory_order_relaxed);
        v.lastOutput = clamp16((sample * v.envelope) >> 15);
        return v.lastOutput;
    }

    void Spu2::step()
    {
        const bool sections = perfLevel() >= 2;
        const uint64_t stepStart = sections ? perfTicks() : 0u;
        int32_t coreOut[2][2] = {};
        for (uint32_t core = 0; core < 2u; ++core)
        {
            const uint32_t base = core * 0x400u;
            const uint32_t mixLeft = reg(base + RegVmixL) | (static_cast<uint32_t>(reg(base + RegVmixL + 2u)) << 16u);
            const uint32_t mixRight = reg(base + RegVmixR) | (static_cast<uint32_t>(reg(base + RegVmixR + 2u)) << 16u);
            int32_t left = 0;
            int32_t right = 0;
            for (uint32_t index = 0; index < VoicesPerCore; ++index)
            {
                const int32_t sample = stepVoice(core, index);
                if (sample == 0)
                    continue;
                const int32_t volLeft = volumeFromRegister(reg(voiceReg(core, index, 0u)));
                const int32_t volRight = volumeFromRegister(reg(voiceReg(core, index, 2u)));
                if ((mixLeft & (1u << index)) != 0u)
                    left += (sample * volLeft) >> 14;
                if ((mixRight & (1u << index)) != 0u)
                    right += (sample * volRight) >> 14;
            }
            coreOut[core][0] = left;
            coreOut[core][1] = right;
        }

        // Salida del nucleo 0 por su volumen maestro, sumada a la seca del nucleo 1 y todo por el volumen maestro del
        // nucleo 1 (simplificacion del enrutado MMIX).
        const auto master = [this](uint32_t core, uint32_t side)
        {
            return volumeFromRegister(reg(RegMvolBase + core * RegMvolStride + side * 2u));
        };
        const uint64_t outputStart = sections ? perfTicks() : 0u;
        std::unique_lock<std::mutex> lock(m_outputMutex);
        const uint64_t locked = sections ? perfTicks() : 0u;
        for (uint32_t side = 0; side < 2u; ++side)
        {
            const int32_t core0 = (coreOut[0][side] * master(0u, side)) >> 14;
            const int32_t mixed = ((core0 + coreOut[1][side]) * master(1u, side)) >> 14;
            m_output.push_back(clamp16(mixed));
        }
        if (m_output.size() > kMaxBufferedFrames * 2u)
        {
            const size_t dropped = m_output.size() / 2u & ~size_t{1};
            m_output.erase(m_output.begin(), m_output.begin() + static_cast<std::ptrdiff_t>(dropped));
            if (perfLevel())
                g_perf.bufferDroppedFrames.fetch_add(dropped / 2u, std::memory_order_relaxed);
        }
        lock.unlock();
        if (sections)
        {
            const uint64_t end = perfTicks();
            g_perf.voiceTicks.fetch_add(outputStart - stepStart, std::memory_order_relaxed);
            g_perf.outputTicks.fetch_add(end - outputStart, std::memory_order_relaxed);
            g_perf.outputLockTicks.fetch_add(locked - outputStart, std::memory_order_relaxed);
            g_perf.stepTicks.fetch_add(end - stepStart, std::memory_order_relaxed);
        }
    }

    void Spu2::advanceTo(uint64_t iopCycle)
    {
        if (iopCycle <= m_lastCycle)
            return;
        const uint64_t samples = (iopCycle - m_lastCycle) / CyclesPerSample;
        const uint64_t start = samples != 0u && perfLevel() ? perfNow() : 0u;
        for (uint64_t i = 0; i < samples; ++i)
            step();
        m_lastCycle += samples * CyclesPerSample;
        if (start != 0u)
        {
            const uint64_t end = perfNow();
            g_perf.stepNs.fetch_add(end - start, std::memory_order_relaxed);
            g_perf.samples.fetch_add(samples, std::memory_order_relaxed);
            g_perf.advanceCalls.fetch_add(1u, std::memory_order_relaxed);
            perfReport(end);
        }
    }

    size_t Spu2::pendingFrames() const
    {
        std::lock_guard<std::mutex> lock(m_outputMutex);
        return m_output.size() / 2u;
    }

    size_t Spu2::drainOutput(int16_t *stereo, size_t maxFrames, size_t maxLatencyFrames)
    {
        const uint64_t start = perfLevel() ? perfNow() : 0u;
        std::lock_guard<std::mutex> lock(m_outputMutex);
        const uint64_t locked = start != 0u ? perfNow() : 0u;
        const size_t available = m_output.size() / 2u;
        size_t dropped = 0u;
        if (maxLatencyFrames != SIZE_MAX && available > maxFrames + maxLatencyFrames)
        {
            dropped = available - maxFrames - maxLatencyFrames;
            m_output.erase(m_output.begin(), m_output.begin() + static_cast<std::ptrdiff_t>(dropped * 2u));
        }
        const size_t frames = std::min(maxFrames, m_output.size() / 2u);
        std::copy_n(m_output.begin(), frames * 2u, stereo);
        m_output.erase(m_output.begin(), m_output.begin() + static_cast<std::ptrdiff_t>(frames * 2u));
        if (start != 0u)
        {
            g_perf.drainNs.fetch_add(perfNow() - start, std::memory_order_relaxed);
            g_perf.drainLockNs.fetch_add(locked - start, std::memory_order_relaxed);
            g_perf.drainCalls.fetch_add(1u, std::memory_order_relaxed);
            g_perf.requestedFrames.fetch_add(maxFrames, std::memory_order_relaxed);
            g_perf.deliveredFrames.fetch_add(frames, std::memory_order_relaxed);
            g_perf.latencyDroppedFrames.fetch_add(dropped, std::memory_order_relaxed);
        }
        return frames;
    }
}
