import socket
import numpy

class OKOMirror():

    def __init__(self,ip):
        self.connection = self.open(ip)


    def open(self,ip):
        TCP_IP = ip
        TCP_PORT = 1234

        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect((TCP_IP, TCP_PORT))
        return connection

    def close(self):
        self.connection.close()

    def set(self, voltages):

        #Import the set voltages and use full length DAC
        full_voltages = numpy.zeros(40,dtype='float64')
        full_voltages[0:6] = voltages[0:6]
        full_voltages[7] = voltages[6]
        full_voltages[12:24] = voltages[7:19]

        #Convert the desired voltages to DAC codes
        DAC = ((2**16-1) * (full_voltages/12.0)).astype('uint16')

        edac40_packet = numpy.zeros(86,dtype='uint8')
        edac40_packet[0:5] = 0xFF
        edac40_packet[5] = 0
        for i in range(0,40):
            edac40_packet[6+2*i] = DAC[i] & 0xFF
            edac40_packet[7+2*i] = (DAC[i]>>8) & 0xFF

        nbytes = self.connection.send(edac40_packet)
