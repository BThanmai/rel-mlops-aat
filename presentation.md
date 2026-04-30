# Presentation Guide — Q-Learning Dynamic Pricing with MLOps

---

# PART 1 — RL (Reinforcement Learning)

## What is this project?
- An AI that learns to set the best product price to maximize revenue
- No price is hardcoded — the agent discovers the optimal price through trial and error
- Uses Q-Learning, a type of Reinforcement Learning

---

## Demand & Price — The key idea

**NOT surge pricing (not Uber)**
- Uber: more demand → higher price — works because people NEED a ride, no choice (inelastic demand)
- Our model: more demand → same optimal price, just MORE revenue because more people buy (elastic demand)
- Think: dress shop. Charge too much → people walk out. Find the sweet spot.

**The two-step reduction (important to understand)**
```
100 customers typed on slider
    ↓  Step 1 — Time multiplier
       × 1.4 if peak  → 140 effective customers
       × 0.7 if off-peak → 70 effective customers
    ↓  Step 2 — Price decides who actually buys
       units_sold = effective_customers × (1 − price_ratio)
    ↓
Revenue = price × units_sold
```

- "35 of 66 customers" on screen = 66 × 0.7 (off-peak) = 46 → then price reduces it to 35
- The "of 66" is the raw input. The 35 comes after BOTH reductions.

**Revenue formula:**
```
base_demand  = customers × 1.4  (peak)   OR  customers × 0.7  (off-peak)
price_ratio  = (price − 50) / 200        →  0 at ₹50 (cheapest), 1 at ₹250 (priciest)
units_sold   = base_demand × (1 − price_ratio)
revenue      = price × units_sold
```

**Mathematical sweet spot = ₹125**
- At ₹50:  lots of buyers, price too low → low revenue
- At ₹125: best balance of price × volume → peak revenue
- At ₹250: barely anyone buys → revenue near 0
- ₹100 and ₹150 give IDENTICAL revenue — both are exactly 25 away from ₹125 (mirror images on curve)

---

## The Revenue Curve (graph on RL tab)
- X-axis: all 21 price options from ₹50 to ₹250
- Y-axis: revenue at that price for the current demand + time
- Shape: always a hill — goes up then comes down
- White dot: where the AI chose to price (near the peak)
- Yellow dot: fixed ₹150 comparison point
- If white dot is higher on the curve than yellow → AI earned more
- Cards below show: `₹price × N orders = ₹revenue` so the math is visible

---

## Every Variable Explained

### Environment (`src/env.py`)
| Variable | What it is |
|---|---|
| `MIN_PRICE = 50` | Cheapest price the AI can set |
| `MAX_PRICE = 250` | Most expensive price the AI can set |
| `NUM_ACTIONS = 21` | 21 price options: ₹50, ₹60, ₹70 … ₹250 |
| `PRICES` | `np.linspace(50, 250, 21)` — the actual price list |
| `self.demand` | Random 1–100 = customers expected today |
| `self.time` | 0 = off-peak, 1 = peak |
| `demand_bin` | demand // 10 → groups demand into 10 buckets (0–9) for Q-table lookup |
| `base_demand` | demand × 1.4 (peak) or × 0.7 (off-peak) |
| `price_ratio` | (price − 50) / 200 → 0 at cheapest, 1 at priciest |
| `units_sold` | base_demand × (1 − price_ratio) |
| `reward` | price × units_sold = revenue earned |

### Agent (`src/agent.py`)
| Variable | What it is |
|---|---|
| `q_table` | 10 × 2 × 21 grid — expected revenue for every (demand bin, time, price) combo |
| `lr = 0.1` | Learning rate — how much each new experience updates the table |
| `gamma = 0.9` | Discount factor — how much future rewards matter vs immediate |
| `epsilon` | Starts 1.0 (fully random) → decays to 0.05 (mostly uses learned policy) |
| `epsilon_decay = 0.9999` | Multiplied each step — slow decay so exploration lasts ~600 episodes |
| `choose_action` | Random price if exploring, best known price if exploiting |
| `update` | Q[state][action] += lr × (reward + gamma × best_future − current_guess) |

### Training (`src/train.py`)
- 5000 episodes × 50 steps = 250,000 total pricing decisions
- Logs reward + epsilon to MLflow every 10% of training
- Saves Q-table to `models/q_table.npy` + a versioned copy in `models/archive/`

### API (`api/app.py`)
- Input: `{ "demand": 66, "time": 0 }`
- Computes: `demand_bin = 66 // 10 = 6`, looks up `q_table[6, 0]`, returns best action
- Output: `{ "price": 120.0, "explanation": { q_values: [...] } }`

---

## One-line summary for teacher
> "The AI tries all 21 prices across 250,000 simulations, remembers which price earned the most revenue for each demand + time state, and uses that to recommend the optimal price in real time."

---
---

# PART 2 — MLOps

## What is MLOps here?
- MLOps = keeping the model tracked, versioned, and automatically improving
- Three things implemented: MLflow (experiment tracking), APScheduler (auto-retrain), Model Versioning

---

## MLflow — Experiment Tracking
- Every training run is automatically logged:
  - **Params**: lr, gamma, epsilon_decay, episodes
  - **Metrics per checkpoint**: avg_reward (per step = total ÷ 50), epsilon value
  - **Final metric**: final_avg_reward (the number shown in the dashboard)
  - **Artifact**: q_table.npy saved as a run artifact
- Stored in `mlruns/` folder locally
- The MLOps tab reads directly from MLflow via `MlflowClient`

**Why divide reward by 50?**
- Training tracks reward summed over 50 steps per episode (~₹200,000)
- Dividing by 50 gives per-decision revenue (~₹4,000) — matches what `evaluate.py` prints
- All numbers in the MLOps dashboard are already divided by 50

---

## APScheduler — Scheduled Retraining
- Runs as a background thread inside Flask when you start `python api/app.py`
- Every 2 minutes: runs `python src/train.py --episodes 1000` as a subprocess
- After it finishes: reloads the new Q-table into memory (live model swap, no restart)
- Frontend countdown shows seconds until next retrain
- Only FINISHED runs are counted in Latest/Best stats — running runs are ignored

---

## Model Versioning
- Every retrain = new MLflow run = new version number (v1, v2, v3 …)
- Saves a timestamped copy: `models/archive/q_table_{run_id[:8]}.npy`
- `models/q_table.npy` = always the latest (what the API uses)
- Version cards show: v# | reward | episodes | timestamp | ✓ done / ● running

---

## MLOps Tab — What each part shows
| Component | What it shows |
|---|---|
| Versions | Total training runs logged (including running ones) |
| Latest Avg Revenue | Per-decision revenue of most recent FINISHED run |
| Best Avg Revenue | Highest per-decision revenue across all FINISHED runs |
| Next Retrain countdown | Seconds until APScheduler fires again |
| Training Curves chart | All runs overlaid — each coloured line = one version, shows reward climbing during training |
| Final Reward Comparison | Bar chart — one bar per version, green = best performing |
| Version cards | Click any card → see that version's epsilon decay curve |
| Epsilon decay chart | Shows exploration (1.0) → exploitation (0.05) transition for that run |

---

## MLOps Endpoints
| Endpoint | What it does |
|---|---|
| `GET /api/mlops` | Returns all versions with status, reward, episodes |
| `GET /api/mlops/run/<id>` | Returns reward curve + epsilon curve for one run |
| `POST /reload` | Reloads latest Q-table without restarting Flask |

---

## MLOps one-line summary for teacher
> "Every 2 minutes the model retrains automatically, each run is versioned and logged in MLflow with reward and epsilon curves, and the dashboard shows all training runs overlaid so you can see improvement across versions."
