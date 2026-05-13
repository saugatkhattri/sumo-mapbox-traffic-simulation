from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import traci
import time
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from ortools.sat.python import cp_model

app = Flask(__name__)

# -------------------------
# USER CONFIGURATION
# -------------------------
# Replace these with your own values before running the project locally.
app.config["SECRET_KEY"] = "YOUR_SECRET_KEY"
MAPBOX_TOKEN = "YOUR_MAPBOX_TOKEN"

# Use "sumo-gui" if you want to see the SUMO simulation window.
SUMO_BINARY = "sumo"
SUMO_CFG_FILE = "kathmandu.sumocfg"

socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

simulation_thread = None
simulation_running = threading.Event()
simulation_running.clear()


def optimize_traffic_cp_sat(lane_vehicles):
    """Optimize green-light duration for lanes based on vehicle counts."""
    if not lane_vehicles:
        return {}

    model = cp_model.CpModel()
    phase_vars = {}

    for lane, count in lane_vehicles.items():
        if count > 0:
            phase_vars[lane] = model.NewIntVar(10, 60, f"green_time_{lane}")

    if not phase_vars:
        return {}

    model.Maximize(sum(phase_vars[lane] * lane_vehicles[lane] for lane in phase_vars))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No optimization solution found.")
        return {}

    return {lane: solver.Value(var) for lane, var in phase_vars.items()}


def build_lane_to_tls_map():
    """Map SUMO lane IDs to their controlling traffic lights."""
    lane_to_tls = {}

    for tls_id in traci.trafficlight.getIDList():
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            for lane in controlled_lanes:
                lane_to_tls[lane] = tls_id
        except traci.TraCIException:
            continue

    return lane_to_tls


def apply_traffic_light_changes(lane_to_tls, optimized_lane_greens):
    """Apply optimized signal timing to traffic lights."""
    for lane_id, duration in optimized_lane_greens.items():
        tls_id = lane_to_tls.get(lane_id)

        if not tls_id:
            continue

        try:
            traci.trafficlight.setPhaseDuration(tls_id, int(duration))
        except traci.TraCIException:
            print(f"Could not update traffic light: {tls_id}")


def sumo_simulation():
    """Run SUMO simulation and send live vehicle data to Mapbox frontend."""
    try:
        traci.start([SUMO_BINARY, "-c", SUMO_CFG_FILE])

        lane_to_tls = build_lane_to_tls_map()
        print(f"Lane-to-traffic-light mappings created: {len(lane_to_tls)}")

        vehicle_counts = defaultdict(int)
        last_reset_time = datetime.now()

        while simulation_running.is_set() and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            vehicles_payload = []

            for veh_id in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(veh_id)
                lon, lat = traci.simulation.convertGeo(x, y)
                angle = traci.vehicle.getAngle(veh_id)

                vehicles_payload.append({
                    "id": veh_id,
                    "x": lon,
                    "y": lat,
                    "angle": angle
                })

                try:
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    vehicle_counts[lane_id] += 1
                except traci.TraCIException:
                    pass

            socketio.emit("update", vehicles_payload)

            now = datetime.now()
            if (now - last_reset_time) >= timedelta(seconds=60):
                optimized = optimize_traffic_cp_sat(dict(vehicle_counts))
                apply_traffic_light_changes(lane_to_tls, optimized)

                vehicle_counts.clear()
                last_reset_time = now

            time.sleep(0.05)

        traci.close()
        socketio.emit("simulation_status", {"status": "stopped"})

    except Exception as error:
        print(f"Simulation error: {error}")
        socketio.emit("simulation_error", {"message": str(error)})

    finally:
        simulation_running.clear()


@socketio.on("connect")
def handle_connect():
    status = "running" if simulation_running.is_set() else "stopped"
    emit("simulation_status", {"status": status})


@socketio.on("start_simulation")
def handle_start():
    global simulation_thread

    if not simulation_running.is_set():
        simulation_running.set()
        simulation_thread = threading.Thread(target=sumo_simulation, daemon=True)
        simulation_thread.start()

        emit("simulation_status", {"status": "running"}, broadcast=True)


@socketio.on("stop_simulation")
def handle_stop():
    global simulation_thread

    if simulation_running.is_set():
        simulation_running.clear()
        emit("simulation_status", {"status": "stopping"}, broadcast=True)

        if simulation_thread and simulation_thread.is_alive():
            simulation_thread.join(timeout=5.0)

        emit("simulation_status", {"status": "stopped"}, broadcast=True)


@app.route("/")
def index():
    return render_template(
        "index.html",
        mapbox_token=MAPBOX_TOKEN
    )


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)