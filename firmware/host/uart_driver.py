import serial
import time
from typing import Optional, List


class UARTDriver:
    FIFO_SIZE = 256

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout
            )
            print(f"UART connected: {port} @ {baud} baud")
        except serial.SerialException as e:
            #raised if port doesn't exist or is alr in use
            print(f"ERROR: Couldn't open port {port}")
            print(F"   {e}")
            print("\nTroubleshooting:")
            print("  - Check that the FPGA is connected")
            print("  - Check the port name (use Device Manager on Windows)")
            print("  - Make sure no other program is using the port")
            raise
    
    #send one byte to chip
    def send_byte(self, data: int) -> None:
        if not 0 <= data <= 255:
            raise ValueError(f"Byte value must be 0-255, got {data}")
        
        #convert int to byte
        byte_data = bytes([data])

        #write to serial port
        written = self.ser.write(byte_data)
        if written != 1:
            raise IOError(f"Failed to write byte, wrote {written} bytes")
        
    #send multiple bytes to chp    
    def send_bytes_to_chip(self, data: bytes) -> None:
        chunk_size = self.FIFO_SIZE//2
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            written = self.ser.write(chunk)
            if written != len(chunk):
                raise IOError(f"Failed to write chunk, wrote {written}/{len(chunk)} bytes")
            time.sleep(0.01)

    #receive 1 byte from chip
    def receive_byte(self) -> Optional[int]:
        data = self.ser.read(1)
        if len(data) == 0:
            return None
        return data[0]
    
    #receive multiple bytes from chip
    def receive_bytes(self, count: int) -> bytes:
        data = self.ser.read(count)
        if len(data) < count:
            print(f"Warning: Only received {len(data)}/{count} bytes (timeout?)")
        return data

    #receive exact number of bytes with overall timeout
    def receive_exact(self, count: int, timeout: Optional[float] = None) -> bytes:
        if count <= 0:
            return b""

        if timeout is None:
            timeout = self.ser.timeout if self.ser.timeout is not None else 1.0

        deadline = time.time() + timeout
        chunks = []
        remaining = count

        while remaining > 0 and time.time() < deadline:
            data = self.ser.read(remaining)
            if data:
                chunks.append(data)
                remaining -= len(data)
            else:
                time.sleep(0.01)

        data = b"".join(chunks)
        if len(data) < count:
            print(f"Warning: Only received {len(data)}/{count} bytes (timeout?)")
        return data

    #discard unread data in RX buffer
    def flush_input(self) -> None:
        self.ser.reset_input_buffer()

    #wait for pending output to be transmitted
    def flush_output(self) -> None:
        self.ser.flush()
    
    #check how many bytes are waiting to be read
    def bytes_waiting(self) -> int:
        return self.ser.in_waiting
    
    #close serial connection
    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"UART closed: {self.port}")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
#test
if __name__ == "__main__":
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    print("="*50)
    print("UART Driver Test")
    print("="*50)
    
    try: 
        with UARTDriver(port, baud=115200) as uart:
            print("\nSending test bytes...")
            test_data = bytes([0x00, 0x55, 0xAA, 0xFF]) 
            uart.send_bytes_to_chip(test_data)
            print(f"Sent: {test_data.hex()}")
            
            print("\nWaiting for response...")
            time.sleep(0.1)
            
            waiting = uart.bytes_waiting()
            print(f"Bytes waiting: {waiting}")
            
            if waiting > 0:
                response = uart.receive_bytes(waiting)
                print(f"Received: {response.hex()}")
            else:
                print("No response (this may be normal if chip doesn't echo)")
            
    except serial.SerialException as e:
        print(f"\nFailed to connect: {e}")
        print("\nMake sure:")
        print(f"  1. FPGA is connected to {port}")
        print("  2. No other program is using the port")
        print("  3. You have permission to access the port")
    
     
