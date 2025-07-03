import live_trading_bot
import http.server
import socketserver
import threading

PORT = 8080 

def serve_qr_code_forever():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving QR code at http://localhost:{PORT}/pionex_login_qr.png")
        httpd.serve_forever()

server_thread = threading.Thread(target=serve_qr_code_forever, daemon=True)
server_thread.start()

trader = live_trading_bot.LiveTrader()
try:
    trader.run()
    data = trader.fetch_live_data()
except KeyboardInterrupt:
    logging.info("\nShutting down trading bot...")
finally:
    trader.cleanup() 