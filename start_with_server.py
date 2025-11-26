import live_trading_bot
import live_trading_bot_direct
import live_trading_bot_multi
import http.server
import socketserver
import threading

# === CONFIGURATION ===
# Set BOT_TYPE to one of: 'single', 'direct', 'multi'
BOT_TYPE = 'multi'  # 'single' for live_trading_bot, 'direct' for live_trading_bot_direct, 'multi' for live_trading_bot_multi

PORT = 8080

# Trading parameters
interval = "15M"
symbol = "RAY_USDT"
symbols = ["RAY_USDT", "XRP_USDT"]  # Used for multi-symbol bot (max 3)

# === QR Code Server ===
def serve_qr_code_forever():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving QR code at http://100.74.60.27:{PORT}/pionex_login_qr.png")
        httpd.serve_forever()

server_thread = threading.Thread(target=serve_qr_code_forever, daemon=True)
server_thread.start()

# === Bot Selection Logic ===
if BOT_TYPE == 'single':
    trader = live_trading_bot.LiveTrader(interval=interval, symbol=symbol)
elif BOT_TYPE == 'direct':
    trader = live_trading_bot_direct.LiveTraderDirect(interval=interval, symbol=symbol)
elif BOT_TYPE == 'multi':
    trader = live_trading_bot_multi.LiveTraderDirect(interval=interval, symbols=symbols)
else:
    raise ValueError(f"Unknown BOT_TYPE: {BOT_TYPE}")

try:
    trader.run()
except KeyboardInterrupt:
    print("\nShutting down trading bot...")
finally:
    trader.cleanup()