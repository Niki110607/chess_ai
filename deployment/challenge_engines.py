import berserk

session = berserk.TokenSession()
client = berserk.Client(session=session)

# Choose A or B:

# --- A. Challenge Lichess's Built-in Stockfish ---
# Levels range from 1 (Easy) to 8 (Hardest)
client.challenges.create_ai(
    level=7,
    clock_limit=180,      # 3 minutes (in seconds)
    clock_increment=2,    # 2 second increment
    color='random'
)
print("Challenged Lichess AI!")

# --- B. Challenge a specific Bot Account ---
# Find a bot on lichess.org/player/bots (e.g., 'maia1' or a Stockfish bot)
"""opponent_username = 'stockfish'

client.challenges.create(
    username=opponent_username,
    rated=False,          # Keep it False for testing!
    clock_limit=600,
    clock_increment=0,
    color='random',
    variant="standard"
)
print(f"Challenged {opponent_username}!")"""