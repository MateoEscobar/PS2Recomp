#pragma once

#include <cstdint>
#include <string>

namespace ps2x::iop
{
    class IopHost;
}

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopMemory;

    // Expand a printf-style format string living in IOP RAM with the MIPS o32 varargs of the
    // current call: arguments 0..3 travel in a0..a3, the rest at sp+16, sp+20, ... (`firstVarArg`
    // is the index of the first variadic argument, 1 for printf(fmt, ...), 2 for fdprintf(fd, fmt,
    // ...)). With `vaListAddress` != 0 the arguments are read from that va_list instead
    // (vfdprintf). Supports %d %i %u %x %X %o %c %s %p %% with flags, width and precision.
    [[nodiscard]] std::string formatIopPrintf(const IopMemory &memory,
                                              const IopCpuState &cpu,
                                              uint32_t formatAddress,
                                              uint32_t firstVarArg,
                                              uint32_t vaListAddress = 0u);

    class IopStdio
    {
    public:
        IopStdio(IopHost &host, IopMemory &memory) noexcept;

        [[nodiscard]] bool dispatchImport(uint16_t ordinal, IopCpuState &cpu);

    private:
        IopHost &m_host;
        IopMemory &m_memory;
    };
}
