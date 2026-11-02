#include "iop_stdio.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <cctype>
#include <cstdio>
#include <algorithm>
#include <string>

namespace ps2x::iop::detail
{
    std::string formatIopPrintf(const IopMemory &memory,
                                const IopCpuState &cpu,
                                uint32_t formatAddress,
                                uint32_t firstVarArg,
                                uint32_t vaListAddress)
    {
        const std::string format = memory.readString(formatAddress, 2048u);
        std::string out;
        out.reserve(format.size() + 32u);

        uint32_t argIndex = 0u;
        const auto nextArg = [&]() -> uint32_t
        {
            uint32_t value = 0u;
            if (vaListAddress != 0u)
            {
                (void)memory.readRam(vaListAddress + argIndex * 4u, &value, sizeof(value));
            }
            else
            {
                const uint32_t slot = firstVarArg + argIndex;
                if (slot < 4u)
                    value = cpu.gpr[4u + slot];
                else
                    (void)memory.readRam(cpu.gpr[29] + slot * 4u, &value, sizeof(value));
            }
            ++argIndex;
            return value;
        };

        for (size_t i = 0u; i < format.size(); ++i)
        {
            const char ch = format[i];
            if (ch != '%')
            {
                out.push_back(ch);
                continue;
            }
            if (i + 1u >= format.size())
            {
                out.push_back('%');
                break;
            }

            // Parse %[flags][width][.precision][length]conversion into a host-side spec.
            std::string spec = "%";
            size_t j = i + 1u;
            while (j < format.size() && (format[j] == '-' || format[j] == '+' || format[j] == ' ' ||
                                         format[j] == '#' || format[j] == '0'))
                spec.push_back(format[j++]);
            if (j < format.size() && format[j] == '*')
            {
                spec += std::to_string(static_cast<int32_t>(nextArg()));
                ++j;
            }
            else
            {
                while (j < format.size() && std::isdigit(static_cast<unsigned char>(format[j])))
                    spec.push_back(format[j++]);
            }
            if (j < format.size() && format[j] == '.')
            {
                spec.push_back(format[j++]);
                if (j < format.size() && format[j] == '*')
                {
                    spec += std::to_string(static_cast<int32_t>(nextArg()));
                    ++j;
                }
                else
                {
                    while (j < format.size() && std::isdigit(static_cast<unsigned char>(format[j])))
                        spec.push_back(format[j++]);
                }
            }
            while (j < format.size() && (format[j] == 'l' || format[j] == 'h' || format[j] == 'z' ||
                                         format[j] == 'L' || format[j] == 'q'))
                ++j; // all IOP integer arguments are 32-bit
            if (j >= format.size())
            {
                out.append(format, i, std::string::npos);
                break;
            }

            const char conversion = format[j];
            char buffer[512];
            int written = 0;
            switch (conversion)
            {
            case 'd':
            case 'i':
                spec.push_back('d');
                written = std::snprintf(buffer, sizeof(buffer), spec.c_str(), static_cast<int32_t>(nextArg()));
                break;
            case 'u':
            case 'x':
            case 'X':
            case 'o':
                spec.push_back(conversion);
                written = std::snprintf(buffer, sizeof(buffer), spec.c_str(), nextArg());
                break;
            case 'c':
                spec.push_back('c');
                written = std::snprintf(buffer, sizeof(buffer), spec.c_str(), static_cast<int>(nextArg() & 0xFFu));
                break;
            case 'p':
                spec += "08x";
                written = std::snprintf(buffer, sizeof(buffer), spec.c_str(), nextArg());
                break;
            case 's':
            {
                const uint32_t address = nextArg();
                const std::string text = address != 0u ? memory.readString(address, 1024u) : std::string("(null)");
                spec.push_back('s');
                written = std::snprintf(buffer, sizeof(buffer), spec.c_str(), text.c_str());
                break;
            }
            case '%':
                buffer[0] = '%';
                written = 1;
                break;
            default:
                // Unknown conversion: keep the raw directive so nothing is silently lost.
                out.append(format, i, j - i + 1u);
                i = j;
                continue;
            }
            if (written > 0)
                out.append(buffer, static_cast<size_t>(std::min<int>(written, static_cast<int>(sizeof(buffer) - 1u))));
            i = j;
        }
        return out;
    }

    IopStdio::IopStdio(IopHost &host, IopMemory &memory) noexcept
        : m_host(host), m_memory(memory)
    {
    }

    bool IopStdio::dispatchImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        const auto logString = [&](std::string_view prefix, uint32_t address, uint32_t resultBias = 0u)
        {
            const std::string text = m_memory.readString(address, 2048u);
            m_host.log(LogLevel::Info, std::string(prefix) + text);
            setV0(static_cast<uint32_t>(text.size()) + resultBias);
        };
        const auto logFormatted = [&](std::string_view prefix, uint32_t formatAddress, uint32_t firstVarArg, uint32_t vaList = 0u)
        {
            const std::string text = formatIopPrintf(m_memory, cpu, formatAddress, firstVarArg, vaList);
            m_host.log(LogLevel::Info, std::string(prefix) + text);
            setV0(static_cast<uint32_t>(text.size()));
        };

        switch (ordinal)
        {
        case 4: // printf(format, ...)
            logFormatted("[IOP printf] ", a0, 1u);
            return true;
        case 5: // getchar
        case 10:
            setV0(0xFFFFFFFFu);
            return true;
        case 6: // putchar
            m_host.log(LogLevel::Info, std::string("[IOP putchar] ") + static_cast<char>(a0 & 0xFFu));
            setV0(a0 & 0xFFu);
            return true;
        case 7: // puts
            logString("[IOP puts] ", a0, 1u);
            return true;
        case 8: // gets
        case 13:
            setV0(0u);
            return true;
        case 9: // fdprintf(fd, format, ...)
            logFormatted("[IOP fdprintf] ", a1, 2u);
            return true;
        case 11:
            setV0(a0 & 0xFFu);
            return true;
        case 12: // fdputs
            logString("[IOP fdputs] ", a0);
            return true;
        case 14: // vfdprintf(fd, format, va_list)
            logFormatted("[IOP vfdprintf] ", a1, 2u, cpu.gpr[6]);
            return true;
        default:
            return false;
        }
    }
}
