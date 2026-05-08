import pandas as pd
import random
import time
import math
import pulp
from copy import deepcopy

DATA_EXCEL_PATH = "60eczaneküçükçekmece.xlsx"
SEGMENT_EXCEL_PATH = "musteri_kumeleme_sonuclari.xlsx"

OUTPUT_TXT = "matheuristic_sonuc.txt"
OUTPUT_EXCEL = "matheuristic_sonuc.xlsx"

DEPOT = "1"
SPEED = 80
SERVICE_TIME_VALUE = 2

SERVICES = ["s1", "s2", "s3", "s4"]

SERVICE_OUT = {
    "s1": 480.0,
    "s2": 660.0,
    "s3": 840.0,
    "s4": 990.0
}

SERVICE_FINISH = {
    "s1": 630.0,
    "s2": 810.0,
    "s3": 960.0,
    "s4": 1440.0
}
SEGMENT_SKORLARI = {
    "Altin": 10.0,
    "Gumus": 2.0,
    "Bronz": 1.0
}

class VRPData:
    def __init__(self, filepath, segment_filepath):
        
        df_talep = pd.read_excel(filepath, sheet_name="talep")
        df_arac = pd.read_excel(filepath, sheet_name="araç")
        df_uzaklik = pd.read_excel(filepath, sheet_name="uzaklıklar", index_col=0)
        df_sure = pd.read_excel(filepath, sheet_name="süreler", index_col=0)
        df_koord = pd.read_excel(filepath, sheet_name="koordinatlar")


        df_seg = pd.read_excel(segment_filepath, sheet_name="Sheet1")

        df_talep.columns = [str(c).strip() for c in df_talep.columns]
        df_arac.columns = [str(c).strip() for c in df_arac.columns]
        df_seg.columns = [str(c).strip() for c in df_seg.columns]
        df_koord.columns = [str(c).strip() for c in df_koord.columns]

        df_koord = df_koord.set_index("eczane")
        df_koord.index = df_koord.index.astype(int)

        df_talep = df_talep.set_index("eczane")
        df_talep.index = df_talep.index.astype(int)

        df_arac["araç"] = df_arac["araç"].astype(str)
        df_arac = df_arac.set_index("araç")

        df_uzaklik.index = df_uzaklik.index.astype(int)
        df_uzaklik.columns = [int(c) for c in df_uzaklik.columns]

        df_sure.index = df_sure.index.astype(int)
        df_sure.columns = [int(c) for c in df_sure.columns]

        df_seg["Store"] = df_seg["Store"].astype(int)
        df_seg = df_seg.set_index("Store")

        node_ids = list(df_talep.index.astype(int))

        self.nodes = [str(i) for i in node_ids]
        self.customers = [n for n in self.nodes if n != DEPOT]
        self.services = SERVICES
        self.vehicles = [str(v) for v in df_arac.index.tolist()]

        self.demand = {
            s: {
                str(i): float(df_talep.loc[i, s])
                for i in node_ids
            }
            for s in self.services
        }

        self.segment = {}
        raw_segment_score = {}

        for i in node_ids:
            node = str(i)

            if i in df_seg.index and "Segment" in df_seg.columns:
                segment_adi = str(df_seg.loc[i, "Segment"]).strip()
            else:
                segment_adi = "Bilinmiyor"

            self.segment[node] = segment_adi

            raw_segment_score[node] = SEGMENT_SKORLARI.get(segment_adi, 1.0)

        max_score = max(raw_segment_score.values())

        self.score = {
            node: raw_segment_score[node] / max_score if max_score > 0 else 1.0
            for node in raw_segment_score
        }

        self.raw_segment_score = raw_segment_score
        self.coordinates = {}
        self.capacity = {
            v: float(df_arac.loc[v, "kapasite"])
            for v in self.vehicles
        }

        self.kmCost = {
            v: float(df_arac.loc[v, "km maliyet"])
            for v in self.vehicles
        }

        self.c = {
            str(i): {
                str(j): float(df_uzaklik.loc[i, j])
                for j in node_ids
            }
            for i in node_ids
        }

        self.time_mat = {
            str(i): {
                str(j): float(df_sure.loc[i, j])
                for j in node_ids
            }
            for i in node_ids
        }

        self.service_time = {
            i: {
                s: SERVICE_TIME_VALUE if i != DEPOT and self.demand[s][i] > 0 else 0.0
                for s in self.services
            }
            for i in self.nodes
        }

        for i in node_ids:
            if i in df_koord.index:

                coords = str(df_koord.loc[i, "Coordinates"]).split(",")

                self.coordinates[str(i)] = {
                    "lat": float(coords[0]),
                    "lon": float(coords[1])
                }

            else:
                self.coordinates[str(i)] = {
                    "lat": 0.0,
                    "lon": 0.0
                }

        print("Koordinat örnekleri:",
              list(self.coordinates.items())[:5])
        print("Okunan düğüm sayısı:", len(self.nodes))
        print("Okunan araçlar:", self.vehicles)
        print("Kapasite:", self.capacity)
        print("kmCost:", self.kmCost)
        print("Segment örnekleri:", list(self.segment.items())[:10])
        print("Ham Segment_Skoru örnekleri:", list(self.raw_segment_score.items())[:10])
        print("Normalize score örnekleri:", list(self.score.items())[:10])


class ALNSSetPartitioning:
    def __init__(self, data, seed=42):
        self.data = data
        random.seed(seed)

    def required_nodes(self, s):
        return [n for n in self.data.customers if self.data.demand[s][n] > 0]

    def route_load(self, route, s):
        return sum(self.data.demand[s][n] for n in route)

    def capacity_ok(self, route, s, v):
        return self.route_load(route, s) <= self.data.capacity[v] + 1e-9

    def compute_times(self, route, s):
        arrival = {}
        late = {}

        current = DEPOT
        current_time = SERVICE_OUT[s]

        for node in route:
            current_time += self.data.service_time[current][s]
            current_time += self.data.time_mat[current][node]

            arrival[node] = current_time
            late[node] = max(0.0, current_time - SERVICE_FINISH[s])

            current = node

        return arrival, late

    def route_distance_cost(self, route, v):
        if not route:
            return 0.0

        total = 0.0
        current = DEPOT

        for node in route:
            total += self.data.c[current][node] * self.data.kmCost[v]
            current = node

        total += self.data.c[current][DEPOT] * self.data.kmCost[v]
        return total

    def route_late_cost(self, route, s):
        _, late = self.compute_times(route, s)

        total = 0.0
        for node in route:
            if self.data.demand[s][node] > 0:
                total += self.data.score[node] * late[node]

        return total

    def route_cost(self, route, s, v):
        return self.route_distance_cost(route, v) + self.route_late_cost(route, s)

    def greedy_initial_solution(self, s):
        routes = {v: [] for v in self.data.vehicles}
        customers = self.required_nodes(s)
        random.shuffle(customers)

        for node in customers:
            best_v = None
            best_route = None
            best_delta = float("inf")

            for v in self.data.vehicles:
                old_route = routes[v]
                old_cost = self.route_cost(old_route, s, v)

                for pos in range(len(old_route) + 1):
                    candidate = old_route[:pos] + [node] + old_route[pos:]

                    if not self.capacity_ok(candidate, s, v):
                        continue

                    new_cost = self.route_cost(candidate, s, v)
                    delta = new_cost - old_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_v = v
                        best_route = candidate

            if best_v is None:
                raise RuntimeError(f"{s} servisinde {node} kapasite nedeniyle yerleşemedi.")

            routes[best_v] = best_route

        return routes

    def destroy_random(self, routes, removal_rate=0.30):
        new_routes = deepcopy(routes)

        all_nodes = []
        for v in self.data.vehicles:
            all_nodes.extend(new_routes[v])

        if not all_nodes:
            return new_routes, []

        k = max(1, int(len(all_nodes) * removal_rate))
        removed = random.sample(all_nodes, min(k, len(all_nodes)))

        for node in removed:
            for v in self.data.vehicles:
                if node in new_routes[v]:
                    new_routes[v].remove(node)

        return new_routes, removed

    def destroy_worst(self, routes, s, removal_rate=0.30):
        new_routes = deepcopy(routes)
        savings = []

        for v in self.data.vehicles:
            route = new_routes[v]
            base_cost = self.route_cost(route, s, v)

            for node in route:
                candidate = [n for n in route if n != node]
                saving = base_cost - self.route_cost(candidate, s, v)
                savings.append((saving, node))

        if not savings:
            return new_routes, []

        savings.sort(reverse=True)
        k = max(1, int(len(savings) * removal_rate))
        removed = [node for _, node in savings[:k]]

        for node in removed:
            for v in self.data.vehicles:
                if node in new_routes[v]:
                    new_routes[v].remove(node)

        return new_routes, removed

    def repair_greedy(self, routes, removed, s):
        new_routes = deepcopy(routes)
        random.shuffle(removed)

        for node in removed:
            best_v = None
            best_route = None
            best_delta = float("inf")

            for v in self.data.vehicles:
                old_route = new_routes[v]
                old_cost = self.route_cost(old_route, s, v)

                for pos in range(len(old_route) + 1):
                    candidate = old_route[:pos] + [node] + old_route[pos:]

                    if not self.capacity_ok(candidate, s, v):
                        continue

                    new_cost = self.route_cost(candidate, s, v)
                    delta = new_cost - old_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_v = v
                        best_route = candidate

            if best_v is None:
                return None

            new_routes[best_v] = best_route

        return new_routes

    def two_opt_route(self, route, s, v):
        if len(route) <= 2:
            return route

        best = route[:]
        best_cost = self.route_cost(best, s, v)
        improved = True

        while improved:
            improved = False

            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    candidate_cost = self.route_cost(candidate, s, v)

                    if candidate_cost + 1e-9 < best_cost:
                        best = candidate
                        best_cost = candidate_cost
                        improved = True

        return best

    def improve_2opt(self, routes, s):
        improved = deepcopy(routes)

        for v in self.data.vehicles:
            improved[v] = self.two_opt_route(improved[v], s, v)

        return improved

    def add_route_to_pool(self, pool, seen, route, s):
        if not route:
            return

        if not any(self.data.demand[s][n] > 0 for n in route):
            return

        key = tuple(route)
        if key in seen:
            return

        feasible_vehicles = [
            v for v in self.data.vehicles
            if self.capacity_ok(route, s, v)
        ]

        if not feasible_vehicles:
            return

        pool.append({
            "path": route[:],
            "demand_set": {n for n in route if self.data.demand[s][n] > 0},
            "feasible_vehicles": feasible_vehicles,
            "costs": {
                v: self.route_cost(route, s, v)
                for v in feasible_vehicles
            }
        })

        seen.add(key)

    def build_route_pool(self, s, iterations=1000):
        pool = []
        seen = set()

        current = self.greedy_initial_solution(s)
        current = self.improve_2opt(current, s)

        for v in self.data.vehicles:
            self.add_route_to_pool(pool, seen, current[v], s)

        current_cost = sum(
            self.route_cost(current[v], s, v)
            for v in self.data.vehicles
        )

        temperature = max(1.0, current_cost * 0.05)
        cooling = 0.995

        for _ in range(iterations):
            if random.random() < 0.5:
                destroyed, removed = self.destroy_random(current, 0.30)
            else:
                destroyed, removed = self.destroy_worst(current, s, 0.30)

            repaired = self.repair_greedy(destroyed, removed, s)

            if repaired is None:
                continue

            repaired = self.improve_2opt(repaired, s)

            candidate_cost = sum(
                self.route_cost(repaired[v], s, v)
                for v in self.data.vehicles
            )

            delta = candidate_cost - current_cost

            if delta < 0:
                accept = True
            else:
                accept = random.random() < (
                    math.exp(-delta / temperature)
                    if temperature > 1e-9
                    else 0
                )

            if accept:
                current = repaired
                current_cost = candidate_cost

            for v in self.data.vehicles:
                self.add_route_to_pool(pool, seen, repaired[v], s)

            temperature *= cooling

        return pool

    def solve_set_partitioning(self, pool, s):
        required = self.required_nodes(s)

        if not required:
            return 0.0, []

        prob = pulp.LpProblem(f"SP_{s}", pulp.LpMinimize)

        var_index = []
        for r_idx, r in enumerate(pool):
            for v in r["feasible_vehicles"]:
                var_index.append((r_idx, v))

        y = pulp.LpVariable.dicts(
            "y",
            (range(len(pool)), self.data.vehicles),
            lowBound=0,
            upBound=1,
            cat="Binary"
        )

        prob += pulp.lpSum(
            pool[r_idx]["costs"][v] * y[r_idx][v]
            for r_idx, v in var_index
        )

        for node in required:
            prob += pulp.lpSum(
                y[r_idx][v]
                for r_idx, v in var_index
                if node in pool[r_idx]["demand_set"]
            ) == 1

        for v in self.data.vehicles:
            prob += pulp.lpSum(
                y[r_idx][vv]
                for r_idx, vv in var_index
                if vv == v
            ) <= 1

        for r_idx, r in enumerate(pool):
            for v in self.data.vehicles:
                if v not in r["feasible_vehicles"]:
                    prob += y[r_idx][v] == 0

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = []
        total = 0.0

        if pulp.LpStatus[prob.status] not in ["Optimal", "Integer Feasible"]:
            print(f"{s} Set Partitioning durumu:", pulp.LpStatus[prob.status])
            return float("inf"), []

        for r_idx, r in enumerate(pool):
            for v in r["feasible_vehicles"]:
                val = y[r_idx][v].value()

                if val is not None and val > 0.5:
                    selected.append({
                        "vehicle": v,
                        "path": r["path"],
                        "cost": r["costs"][v],
                        "distance_cost": self.route_distance_cost(r["path"], v),
                        "late_cost": self.route_late_cost(r["path"], s)
                    })

                    total += r["costs"][v]

        return total, selected

    def solve(self, iterations=1000):
        total_z = 0.0
        all_selected = {}

        print("ALNS + Set Partitioning Matheuristic başlıyor.")

        for s in self.data.services:
            print(f"\n--- {s} ---")

            pool = self.build_route_pool(s, iterations)
            print(f"Rota havuzu büyüklüğü: {len(pool)}")

            cost, selected = self.solve_set_partitioning(pool, s)

            total_z += cost
            all_selected[s] = selected

            for sel in selected:
                route_text = "1->" + "->".join(sel["path"]) + "->1"
                print(
                    f"Araç: {sel['vehicle']} | "
                    f"Rota: {route_text} | "
                    f"Mesafe: {sel['distance_cost']:.2f} | "
                    f"Late: {sel['late_cost']:.2f} | "
                    f"Toplam: {sel['cost']:.2f}"
                )

            print(f"{s} maliyet: {cost:.4f}")

        print("\n==============================")
        print(f"TOPLAM MATHEURISTIC Z = {total_z:.4f}")
        print("==============================")

        write_outputs(all_selected, total_z, self.data)

        frontend_routes = []

        for s, selected_routes in all_selected.items():

            for sel in selected_routes:

                vehicle = sel["vehicle"]
                route = sel["path"]

                coords = []

                # depo başlangıç
                coords.append({
                    "id": DEPOT,
                    "lat": self.data.coordinates[DEPOT]["lat"],
                    "lng": self.data.coordinates[DEPOT]["lon"]
                })

                # rota
                for node in route:

                    coords.append({
                        "id": node,
                        "lat": self.data.coordinates[node]["lat"],
                        "lng": self.data.coordinates[node]["lon"]
                    })

                # depo dönüş
                coords.append({
                    "id": DEPOT,
                    "lat": self.data.coordinates[DEPOT]["lat"],
                    "lng": self.data.coordinates[DEPOT]["lon"]
                })

                frontend_routes.append({
                    "service": s,
                    "vehicle": vehicle,
                    "route": ["1"] + route + ["1"],
                    "coordinates": coords,
                    "distance_cost": sel["distance_cost"],
                    "late_cost": sel["late_cost"],
                    "total_cost": sel["cost"]
                })

        return {
            "total_cost": total_z,
            "routes": frontend_routes
        }


def write_outputs(all_selected, total_z, data):
    rows = []

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("ALNS + SET PARTITIONING MATHEURISTIC SONUCU\n")
        f.write("=" * 60 + "\n\n")
        f.write("Amaç Fonksiyonu: Mesafe maliyeti + score * Late\n")
        f.write(f"Toplam Z = {total_z:.4f}\n\n")

        for s, selected_routes in all_selected.items():
            service_total = 0.0

            f.write(f"Servis: {s}\n")
            f.write("-" * 60 + "\n")

            for sel in selected_routes:
                vehicle = sel["vehicle"]
                route = sel["path"]
                route_text = "1->" + "->".join(route) + "->1"

                distance_cost = sel["distance_cost"]
                late_cost = sel["late_cost"]
                total_cost = sel["cost"]

                service_total += total_cost

                f.write(f"Araç: {vehicle}\n")
                f.write(f"Rota: {route_text}\n")
                f.write(f"Mesafe maliyeti: {distance_cost:.4f}\n")
                f.write(f"Score*Late maliyeti: {late_cost:.4f}\n")
                f.write(f"Toplam rota maliyeti: {total_cost:.4f}\n\n")

                route_coordinates = []

                # depo başlangıç
                if DEPOT in data.coordinates:
                    route_coordinates.append((
                        data.coordinates[DEPOT]["lat"],
                        data.coordinates[DEPOT]["lon"]
                    ))

                # rota düğümleri
                for node in route:
                    if node in data.coordinates:
                        route_coordinates.append((
                            data.coordinates[node]["lat"],
                            data.coordinates[node]["lon"]
                        ))

                # depo dönüş
                if DEPOT in data.coordinates:
                    route_coordinates.append((
                        data.coordinates[DEPOT]["lat"],
                        data.coordinates[DEPOT]["lon"]
                    ))

                rows.append({
                    "Servis": s,
                    "Araç": vehicle,
                    "Rota": route_text,
                    "Koordinatlar": str(route_coordinates),
                    "Mesafe_Maliyeti": distance_cost,
                    "Score_Late_Maliyeti": late_cost,
                    "Toplam_Rota_Maliyeti": total_cost
                })

            f.write(f"{s} toplam maliyet: {service_total:.4f}\n\n")

        f.write("=" * 60 + "\n")
        f.write(f"GENEL TOPLAM Z = {total_z:.4f}\n")

    pd.DataFrame(rows).to_excel(OUTPUT_EXCEL, index=False)

    print(f"\nTXT oluşturuldu: {OUTPUT_TXT}")
    print(f"Excel oluşturuldu: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    start = time.time()

    data = VRPData(
        filepath=DATA_EXCEL_PATH,
        segment_filepath=SEGMENT_EXCEL_PATH
    )

    solver = ALNSSetPartitioning(
        data=data,
        seed=42
    )

    solver.solve(iterations=2000)

    print(f"\nÇözüm süresi: {time.time() - start:.2f} saniye")