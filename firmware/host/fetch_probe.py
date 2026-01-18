from uart_driver import UARTDriver
from program_loader import ProgramLoader


def main():
    import argparse

    parser = argparse.ArgumentParser(description="uTPU UART fetch probe")
    parser.add_argument("--port", "-p", required=True, help="Serial port, e.g. /dev/ttyUSB1")
    parser.add_argument("--addr", "-a", type=lambda x: int(x, 0), default=0x000, help="Base address (hex or dec)")
    parser.add_argument("--count", "-c", type=int, default=4, help="Number of int4 values to read")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    uart = UARTDriver(args.port, baud=115200)
    loader = ProgramLoader(uart, verbose=args.verbose)
    loader.resetChip()

    print(f"Fetching {args.count} values from 0x{args.addr:03X}...")
    values = loader.readResults(args.addr, args.count)
    print("Results:", values)

    uart.close()


if __name__ == "__main__":
    main()
