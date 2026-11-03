// SPU2 emulation tests (src/emulator/core/spu2.*).
// Cases ported from god-of-war-recomp (GPLv3, ps2xTest/src/ps2_spu2_tests.cpp; the NAX case was
// adapted there from Waffle565/God-Of-War-Native-Port runtime@65e1d79 with the author's permission),
// plus a The Warriors case: an IRX polling NAX from synchronous guest code must see it advance.
#include "iop_compat_test_support.h"
#include "emulator/core/iop_memory.h"
#include "emulator/core/spu2.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <vector>

using ps2x::iop::detail::IopMemory;
using ps2x::iop::detail::Spu2;

namespace
{
    constexpr uint32_t kBase = Spu2::RegBase;

    void check(bool condition, const char *message)
    {
        iop_test::require(condition, message);
    }

    void setAddr(Spu2 &spu, uint32_t offset, uint32_t halfwords)
    {
        spu.write16(kBase + offset, static_cast<uint16_t>(halfwords >> 16u));
        spu.write16(kBase + offset + 2u, static_cast<uint16_t>(halfwords));
    }

    // 16-byte ADPCM block: header (shift, filter, flags) and 28 nibbles.
    void writeBlock(Spu2 &spu, uint32_t address, uint8_t shift, uint8_t filter, uint8_t flags,
                    const std::vector<uint8_t> &nibbles)
    {
        spu.writeRam(address, static_cast<uint16_t>(shift | (filter << 4u) | (flags << 8u)));
        for (uint32_t word = 0; word < 7u; ++word)
        {
            uint16_t packed = 0;
            for (uint32_t n = 0; n < 4u; ++n)
            {
                const size_t i = word * 4u + n;
                packed |= static_cast<uint16_t>((i < nibbles.size() ? nibbles[i] : 0u) & 0xFu) << (n * 4u);
            }
            spu.writeRam(address + 1u + word, packed);
        }
    }

    // Core 0 voice 0: SSA, 48 kHz pitch, instant attack and full sustain, mixed to both sides.
    void setupVoice0(Spu2 &spu, uint32_t ssa)
    {
        setAddr(spu, Spu2::RegVoiceAddr + 0u, ssa); // SSA
        spu.write16(kBase + 0x4u, 0x1000u);          // PITCH
        spu.write16(kBase + 0x6u, 0x000Fu);          // ADSR1: attack 0 (fastest), sustain 0x8000
        spu.write16(kBase + 0x8u, 0x0000u);          // ADSR2
        spu.write16(kBase + 0x0u, 0x3FFFu);          // VOLL
        spu.write16(kBase + 0x2u, 0x3FFFu);          // VOLR
        spu.write16(kBase + Spu2::RegVmixL, 0x0001u);
        spu.write16(kBase + Spu2::RegVmixR, 0x0001u);
        for (uint32_t core = 0; core < 2u; ++core)
        {
            spu.write16(kBase + Spu2::RegMvolBase + core * Spu2::RegMvolStride, 0x3FFFu);
            spu.write16(kBase + Spu2::RegMvolBase + core * Spu2::RegMvolStride + 2u, 0x3FFFu);
        }
    }

    void manualTransfer()
    {
        Spu2 spu;
        setAddr(spu, Spu2::RegTsa, 0x1234u);
        spu.write16(kBase + Spu2::RegData, 0xAAAAu);
        spu.write16(kBase + Spu2::RegData, 0xBBBBu);
        check(spu.ram(0x1234u) == 0xAAAAu, "first halfword at TSA");
        check(spu.ram(0x1235u) == 0xBBBBu, "second halfword after TSA advanced");
        check(spu.read16(kBase + Spu2::RegTsa + 2u) == 0x1236u, "TSA should advance by two");
    }

    void iopWritesReachDataPortOnce()
    {
        IopMemory memory;
        memory.write16(kBase + Spu2::RegTsa, 0u);
        memory.write16(kBase + Spu2::RegTsa + 2u, 0x100u);
        memory.write16(kBase + Spu2::RegData, 0x1111u);
        memory.write16(kBase + Spu2::RegData, 0x2222u);
        check(memory.spu2().ram(0x100u) == 0x1111u, "first data port write");
        check(memory.spu2().ram(0x101u) == 0x2222u, "second data port write");
        check(memory.read16(kBase + Spu2::RegTsa + 2u) == 0x102u,
              "each 16-bit write should advance TSA once, not once per byte");
    }

    void dmaCopiesIopRam()
    {
        IopMemory memory;
        const uint8_t payload[16] = {0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
                                     0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
        check(memory.writeRam(0x80000u, payload, sizeof(payload)), "payload should be written to IOP RAM");
        memory.write16(kBase + 0x400u + Spu2::RegTsa, 0u); // core 1
        memory.write16(kBase + 0x400u + Spu2::RegTsa + 2u, 0x2000u);
        memory.write32(0x1F801500u, 0x80000u);         // MADR channel 7
        memory.write32(0x1F801504u, (1u << 16u) | 4u); // BCR: 1 block of 4 words
        memory.write32(0x1F801508u, 0x01000201u);      // CHCR: start, memory to SPU2
        check(memory.spu2().ram(0x2000u) == 0x3210u, "first halfword copied");
        check(memory.spu2().ram(0x2007u) == 0xEFCDu, "last halfword copied");
        check(memory.read16(kBase + 0x400u + Spu2::RegTsa + 2u) == 0x2008u, "DMA should advance the core TSA");
        check((memory.read16(kBase + 0x400u + Spu2::RegStatx) & Spu2::StatxDmaReady) != 0u,
              "STATX should report the transfer as done, as before");
        const auto start = memory.takeDmaStart();
        check(start.has_value() && start->irq == 0x28, "the core 1 DMA interrupt should still be scheduled");
    }

    void adpcmDecode()
    {
        Spu2 spu;
        writeBlock(spu, 0x1000u, 12u, 0u, 0u, {1u, 2u, 3u, 0xFu});
        writeBlock(spu, 0x1008u, 0u, 1u, 0u, {1u, 0u});
        setupVoice0(spu, 0x1000u);
        spu.write16(kBase + Spu2::RegKon, 0x0001u);
        spu.step();
        const auto &v = spu.voice(0u, 0u);
        check(v.decoded[0] == 1, "shift 12 keeps the nibble value");
        check(v.decoded[2] == 3, "third nibble");
        check(v.decoded[3] == -1, "nibbles are signed");
        for (int i = 0; i < 28; ++i)
            spu.step();
        check(spu.voice(0u, 0u).decoded[0] == 4096, "shift 0 scales the nibble by 4096");
        check(spu.voice(0u, 0u).decoded[1] == 3840, "filter 1 adds 60/64 of the previous sample");
    }

    void loopEndFlags()
    {
        Spu2 spu;
        writeBlock(spu, 0x2000u, 0u, 0u, 0x01u, {1u}); // end, no repeat
        writeBlock(spu, 0x3000u, 0u, 0u, 0x07u, {1u}); // loop start + end + repeat
        setupVoice0(spu, 0x2000u);
        setAddr(spu, Spu2::RegVoiceAddr + 0xCu, 0x3000u); // voice 1 SSA
        spu.write16(kBase + 0x10u + 0x4u, 0x1000u);
        spu.write16(kBase + Spu2::RegKon, 0x0003u);
        for (int i = 0; i < 40; ++i)
            spu.step();
        check(!spu.voice(0u, 0u).active, "a block ending without repeat should silence the voice");
        check(spu.voice(0u, 1u).active, "a repeating block should keep the voice playing");
        check(spu.voice(0u, 1u).nax == 0x3008u, "the repeating voice should have jumped back to its loop start");
        check(spu.read16(kBase + Spu2::RegEndx) == 0x0003u, "both voices should report the end flag in ENDX");
        spu.write16(kBase + Spu2::RegEndx, 0u);
        check(spu.read16(kBase + Spu2::RegEndx) == 0u, "writing ENDX should clear it");
    }

    // Streams that poll NAX abort when it falls outside both halves of their ring: NAX must stay
    // inside the playing block, never in the next one.
    void naxInsidePlayingBlock()
    {
        Spu2 spu;
        writeBlock(spu, 0x6000u, 0u, 0u, 0x04u, std::vector<uint8_t>(28u, 1u));
        writeBlock(spu, 0x6008u, 0u, 0u, 0x03u, std::vector<uint8_t>(28u, 1u));
        setupVoice0(spu, 0x6000u);
        spu.write16(kBase + Spu2::RegKon, 0x0001u);
        const auto nax = [&]
        {
            return (static_cast<uint32_t>(spu.read16(kBase + Spu2::RegVoiceAddr + 8u)) << 16u) |
                   spu.read16(kBase + Spu2::RegVoiceAddr + 10u);
        };
        check(nax() == 0x6001u, "before decoding, NAX reads the start block | 1");
        size_t outside = 0;
        for (int i = 0; i < 28 * 6; ++i)
        {
            spu.step();
            const uint32_t value = nax();
            const bool inFirst = value >= 0x6001u && value <= 0x6007u;
            const bool inSecond = value >= 0x6009u && value <= 0x600Fu;
            if (!inFirst && !inSecond)
                ++outside;
        }
        check(outside == 0u, "NAX must stay inside the playing block of the ring");
    }

    void adsrAttackRelease()
    {
        Spu2 spu;
        writeBlock(spu, 0x4000u, 0u, 0u, 0x03u, {7u, 7u, 7u, 7u});
        setupVoice0(spu, 0x4000u);
        spu.write16(kBase + 0x8u, 0x0000u); // release rate 0 (fastest)
        spu.write16(kBase + Spu2::RegKon, 0x0001u);
        for (int i = 0; i < 3; ++i) // attack 0: +0x3800 per sample, maximum on the third
            spu.step();
        check(spu.read16(kBase + 0xAu) == 0x7FFFu, "ENVX should reach the maximum quickly");
        spu.write16(kBase + Spu2::RegKoff, 0x0001u);
        for (int i = 0; i < 16; ++i)
            spu.step();
        check(spu.read16(kBase + 0xAu) == 0u, "ENVX should reach zero after release");
        check(!spu.voice(0u, 0u).active, "the voice should be off after its release");
    }

    void voiceIsMixed()
    {
        Spu2 spu;
        writeBlock(spu, 0x5000u, 0u, 0u, 0x03u, std::vector<uint8_t>(28u, 3u));
        setupVoice0(spu, 0x5000u);
        spu.write16(kBase + 0x2u, 0x4000u); // VOLR = -1.0
        spu.write16(kBase + Spu2::RegKon, 0x0001u);
        spu.advanceTo(Spu2::CyclesPerSample * 64u);
        check(spu.pendingFrames() == 64u, "one frame per 768 IOP cycles");
        std::vector<int16_t> out(128u);
        check(spu.drainOutput(out.data(), 64u) == 64u, "all frames should drain");
        check(out[126] > 0, "left output should follow the positive samples");
        check(out[127] < 0, "right output should follow the negative voice volume");
        check(spu.pendingFrames() == 0u, "drained frames should be removed");
    }

    // Reference of the observable output: several voices on both cores with pitches, ADSR, loops and
    // KON/KOFF midway; the FNV-1a hash covers every stereo sample and the final ENVX. A performance
    // change of the SPU2 must keep it; an intended emulation change updates it with the reason.
    void multiVoiceReferenceHash()
    {
        Spu2 spu;
        uint32_t seed = 0x12345678u;
        const auto next = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8u; };
        for (uint32_t block = 0; block < 64u; ++block)
        {
            std::vector<uint8_t> nibbles(28u);
            for (auto &nibble : nibbles)
                nibble = static_cast<uint8_t>(next() & 0xFu);
            const uint8_t flags = (block % 8u == 0u) ? 0x04u : (block % 8u == 7u) ? 0x03u : 0x00u;
            writeBlock(spu, 0x10000u + block * 8u, static_cast<uint8_t>(next() % 13u),
                       static_cast<uint8_t>(next() % 5u), flags, nibbles);
        }
        constexpr uint16_t pitches[6] = {0x0400u, 0x0800u, 0x1000u, 0x1555u, 0x2000u, 0x3FFFu};
        constexpr uint16_t adsr1[6] = {0x000Fu, 0x80FFu, 0x3F48u, 0x0A03u, 0xFF00u, 0x1F7Fu};
        constexpr uint16_t adsr2[6] = {0x0000u, 0x5FC5u, 0x1F0Au, 0xC0E3u, 0x4040u, 0x9F7Fu};
        for (uint32_t core = 0; core < 2u; ++core)
        {
            const uint32_t base = core * 0x400u;
            for (uint32_t v = 0; v < 6u; ++v)
            {
                const uint32_t voice = v * 3u + core;
                setAddr(spu, base + Spu2::RegVoiceAddr + voice * 0xCu, 0x10000u + ((v * 5u + core * 3u) % 8u) * 64u);
                spu.write16(kBase + base + voice * 0x10u + 0x0u, static_cast<uint16_t>(0x1000u + v * 0x0400u));
                spu.write16(kBase + base + voice * 0x10u + 0x2u, static_cast<uint16_t>(0x3FFFu - v * 0x0500u));
                spu.write16(kBase + base + voice * 0x10u + 0x4u, pitches[(v + core) % 6u]);
                spu.write16(kBase + base + voice * 0x10u + 0x6u, adsr1[v]);
                spu.write16(kBase + base + voice * 0x10u + 0x8u, adsr2[(v + core) % 6u]);
            }
            spu.write16(kBase + base + Spu2::RegVmixL, 0xFFFFu);
            spu.write16(kBase + base + Spu2::RegVmixL + 2u, 0x00FFu);
            spu.write16(kBase + base + Spu2::RegVmixR, 0xAAAAu);
            spu.write16(kBase + base + Spu2::RegVmixR + 2u, 0x0055u);
            spu.write16(kBase + Spu2::RegMvolBase + core * Spu2::RegMvolStride, 0x3000u);
            spu.write16(kBase + Spu2::RegMvolBase + core * Spu2::RegMvolStride + 2u, 0x2800u);
        }

        uint64_t hash = 0xCBF29CE484222325ull;
        const auto mix = [&hash](uint16_t value)
        {
            for (uint32_t shift = 0; shift < 16u; shift += 8u)
            {
                hash ^= static_cast<uint8_t>(value >> shift);
                hash *= 0x100000001B3ull;
            }
        };
        std::vector<int16_t> out(2u * 4096u);
        uint64_t cycle = 0;
        size_t frames = 0;
        size_t audible = 0;
        for (uint32_t chunk = 0; chunk < 64u; ++chunk)
        {
            if (chunk == 0u)
            {
                spu.write16(kBase + Spu2::RegKon, 0x0249u);
                spu.write16(kBase + 0x400u + Spu2::RegKon, 0x0492u);
            }
            if (chunk == 12u)
            {
                spu.write16(kBase + Spu2::RegKon + 2u, 0x0001u);
                spu.write16(kBase + 0x400u + Spu2::RegKon, 0x8000u);
            }
            if (chunk == 40u)
            {
                spu.write16(kBase + Spu2::RegKoff, 0x0049u);
                spu.write16(kBase + 0x400u + Spu2::RegKoff, 0x0402u);
            }
            cycle += Spu2::CyclesPerSample * (37u + next() % 160u) + next() % Spu2::CyclesPerSample;
            spu.advanceTo(cycle);
            const size_t drained = spu.drainOutput(out.data(), 4096u);
            frames += drained;
            for (size_t i = 0; i < drained * 2u; ++i)
            {
                mix(static_cast<uint16_t>(out[i]));
                audible += out[i] != 0 ? 1u : 0u;
            }
        }
        for (uint32_t core = 0; core < 2u; ++core)
            for (uint32_t voice = 0; voice < Spu2::VoicesPerCore; ++voice)
                mix(spu.read16(kBase + core * 0x400u + voice * 0x10u + 0xAu));

        check(frames > 6000u, "the scenario should produce several thousand frames");
        check(audible > frames / 2u, "most of the reference output should be audible, not silence");
        if (hash != 0xa97415b49fd7d37cull)
        {
            std::fprintf(stderr, "SPU2 reference hash changed: 0x%016llx\n", static_cast<unsigned long long>(hash));
            check(false, "SPU2 reference hash changed");
        }
    }

    void latencyLimitDropsOldest()
    {
        Spu2 spu;
        for (int i = 0; i < 100; ++i)
            spu.step();
        std::vector<int16_t> out(20u);
        check(spu.drainOutput(out.data(), 10u, 30u) == 10u, "ten frames should be returned");
        check(spu.pendingFrames() == 30u, "frames beyond the latency limit should be discarded, keeping the newest");
    }

    void idleIopProducesFrames()
    {
        iop_test::Host host;
        ps2x::iop::IopSubsystem subsystem(host);
        subsystem.runEeCycles(8u * Spu2::CyclesPerSample * 100u); // 8 EE cycles per IOP cycle
        std::vector<int16_t> out(400u);
        check(subsystem.drainAudio(out.data(), 200u) == 100u,
              "an idle IOP should still produce audio frames up to the last cycle run");
    }

    void irqaRaisesInterrupt()
    {
        Spu2 spu;
        setAddr(spu, Spu2::RegIrqa, 0x6002u);
        spu.write16(kBase + Spu2::RegAttr, 0x8040u);
        setAddr(spu, Spu2::RegTsa, 0x6000u);
        spu.write16(kBase + Spu2::RegData, 1u);
        check(!spu.takeInterrupt(), "no IRQ before IRQA");
        spu.write16(kBase + Spu2::RegData, 2u);
        spu.write16(kBase + Spu2::RegData, 3u);
        check(spu.takeInterrupt(), "writing IRQA should raise the IRQ");
        check((spu.read16(kBase + 0x7C2u) & 0x4u) != 0u, "SPDIF_IRQINFO should flag core 0");
        check(!spu.takeInterrupt(), "the IRQ should be consumed once");
    }

#pragma pack(push, 1)
    struct ElfHeader
    {
        uint8_t ident[16];
        uint16_t type;
        uint16_t machine;
        uint32_t version;
        uint32_t entry;
        uint32_t phoff;
        uint32_t shoff;
        uint32_t flags;
        uint16_t ehsize;
        uint16_t phentsize;
        uint16_t phnum;
        uint16_t shentsize;
        uint16_t shnum;
        uint16_t shstrndx;
    };

    struct ProgramHeader
    {
        uint32_t type;
        uint32_t offset;
        uint32_t vaddr;
        uint32_t paddr;
        uint32_t filesz;
        uint32_t memsz;
        uint32_t flags;
        uint32_t align;
    };
#pragma pack(pop)

    // The Warriors: MultiStream (IOP.IRX) polls the NAX of its streaming voices to refill the
    // rings, partly from synchronous guest code (RPC handlers run through callFunction). Here a
    // module entry point keys on core 0 voice 0 and spins reading NAX: it returns 7 when NAX
    // moves and 99 when the bounded loop runs out. Without the SPU2 advancing inside guest code,
    // NAX never moves while the entry point runs.
    void naxAdvancesWhileGuestCodePolls()
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        const uint32_t code[] = {
            0x3C08BF90u, // lui   t0, 0xbf90          (SPU2 registers, KSEG1)
            0x34090100u, // ori   t1, zero, 0x100
            0xA50001C0u, // sh    zero, 0x1c0(t0)     SSA hi
            0xA50901C2u, // sh    t1, 0x1c2(t0)       SSA lo = 0x100
            0x34091000u, // ori   t1, zero, 0x1000
            0xA5090004u, // sh    t1, 0x4(t0)         PITCH = 48 kHz
            0x34090001u, // ori   t1, zero, 1
            0xA50901A0u, // sh    t1, 0x1a0(t0)       KON voice 0
            0x950A01CAu, // lhu   t2, 0x1ca(t0)       initial NAX lo
            0x3C0B0003u, // lui   t3, 3               loop bound 0x30000 (also fills t2's load delay)
            0x950C01CAu, // loop: lhu t4, 0x1ca(t0)
            0x00000000u, // nop                       (R3000 load delay slot)
            0x158A0005u, // bne   t4, t2, changed
            0x256BFFFFu, // addiu t3, t3, -1          (delay slot)
            0x1D60FFFBu, // bgtz  t3, loop
            0x00000000u, // nop
            0x03E00008u, // jr    ra
            0x24020063u, // addiu v0, zero, 99        (delay slot)
            0x03E00008u, // changed: jr ra
            0x24020007u, // addiu v0, zero, 7         (delay slot)
        };

        ElfHeader header{};
        header.ident[0] = 0x7Fu;
        header.ident[1] = 'E';
        header.ident[2] = 'L';
        header.ident[3] = 'F';
        header.ident[4] = 1u;
        header.ident[5] = 1u;
        header.ident[6] = 1u;
        header.type = 2u;
        header.machine = 8u;
        header.version = 1u;
        header.entry = loadAddress;
        header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader);
        header.phentsize = sizeof(ProgramHeader);
        header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u;
        program.offset = codeOffset;
        program.vaddr = loadAddress;
        program.paddr = loadAddress;
        program.filesz = sizeof(code);
        program.memsz = sizeof(code);
        program.flags = 5u;
        program.align = 4u;

        iop_test::Host host;
        constexpr uint32_t image = 0x100u;
        check(host.writeGuest(image, &header, sizeof(header)), "ELF header should fit in guest RAM");
        check(host.writeGuest(image + sizeof(header), &program, sizeof(program)), "program header should fit");
        check(host.writeGuest(image + codeOffset, code, sizeof(code)), "code should fit");

        ps2x::iop::IopSubsystem subsystem(host);
        const ps2x::iop::ModuleLoadResult result = subsystem.loadModuleBuffer(image);
        check(result.handled, "the emulator should claim the IRX");
        check(result.startResult == 7, "NAX should advance while guest code polls it");
    }
}

int main()
{
    const struct
    {
        const char *name;
        void (*run)();
    } cases[] = {
        {"manual transfer writes SPU RAM at TSA and advances it", manualTransfer},
        {"IOP 16-bit writes reach the SPU2 data port exactly once", iopWritesReachDataPortOnce},
        {"SPU DMA copies IOP RAM into SPU RAM and keeps STATX and IRQ timing", dmaCopiesIopRam},
        {"ADPCM decodes shift and prediction filters", adpcmDecode},
        {"loop end flags stop or repeat the voice and set ENDX", loopEndFlags},
        {"NAX reads inside the playing block, never the next one", naxInsidePlayingBlock},
        {"ADSR attacks to full level and releases to silence after KOFF", adsrAttackRelease},
        {"an active voice is mixed into the stereo output", voiceIsMixed},
        {"multi-voice mix keeps its reference output hash", multiVoiceReferenceHash},
        {"draining with a latency limit drops the oldest frames", latencyLimitDropsOldest},
        {"the IOP subsystem exposes one SPU2 frame per 768 IOP cycles", idleIopProducesFrames},
        {"reaching IRQA with IRQs enabled raises the SPU2 interrupt", irqaRaisesInterrupt},
        {"NAX advances while synchronous guest code polls it", naxAdvancesWhileGuestCodePolls},
    };
    int failures = 0;
    for (const auto &test : cases)
    {
        try
        {
            test.run();
            std::cout << "[ OK ] " << test.name << '\n';
        }
        catch (const std::exception &error)
        {
            ++failures;
            std::cout << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}
