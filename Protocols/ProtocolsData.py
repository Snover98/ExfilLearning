from Protocols.ProtocolClasses import Layer4Protocol, ProtocolEnum

textual_protocols = {
    # SMTP
    Layer4Protocol(ProtocolEnum.TCP, 25),   Layer4Protocol(ProtocolEnum.UDP, 25),
    Layer4Protocol(ProtocolEnum.TCP, 587),  Layer4Protocol(ProtocolEnum.UDP, 587),
    Layer4Protocol(ProtocolEnum.TCP, 465),  Layer4Protocol(ProtocolEnum.UDP, 465),
    # HTTP
    Layer4Protocol(ProtocolEnum.TCP, 80),   Layer4Protocol(ProtocolEnum.UDP, 80),
    Layer4Protocol(ProtocolEnum.TCP, 8080), Layer4Protocol(ProtocolEnum.UDP, 8080),
    # IMAP
    Layer4Protocol(ProtocolEnum.TCP, 143),  Layer4Protocol(ProtocolEnum.UDP, 143),
    # POP
    Layer4Protocol(ProtocolEnum.TCP, 109),  Layer4Protocol(ProtocolEnum.UDP, 110),
    Layer4Protocol(ProtocolEnum.TCP, 110),  Layer4Protocol(ProtocolEnum.UDP, 109),
    Layer4Protocol(ProtocolEnum.TCP, 995),  Layer4Protocol(ProtocolEnum.UDP, 995),
    # FTP
    Layer4Protocol(ProtocolEnum.TCP, 20),   Layer4Protocol(ProtocolEnum.UDP, 20),  # can also be binary
    Layer4Protocol(ProtocolEnum.TCP, 21),   Layer4Protocol(ProtocolEnum.UDP, 21)
}
