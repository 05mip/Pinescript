import live_trading_bot
import http.server
import socketserver
import threading

PORT = 8080

def serve_qr_code_forever():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving QR code at http://100.74.60.27:{PORT}/pionex_login_qr.png")
        httpd.serve_forever()

server_thread = threading.Thread(target=serve_qr_code_forever, daemon=True)
server_thread.start()

interval = "15M"
symbol = "RAY_USDT"

trader = live_trading_bot.LiveTrader(interval=interval, symbol=symbol)
try:
    trader.run()
except KeyboardInterrupt:
    print("\nShutting down trading bot...")
finally:
    trader.cleanup()