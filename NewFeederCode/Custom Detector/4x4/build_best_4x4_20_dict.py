
import cv2
import numpy as np
import itertools
import networkx as nx

SOURCE_DICT = cv2.aruco.DICT_4X4_50
TARGET_COUNT = 20
OUT_TXT = "best_4x4_20_markers.txt"

def rot_hamming(a, b):
    return min(int(np.count_nonzero(a != np.rot90(b, k))) for k in range(4))

def self_rot_distance(a):
    return min(int(np.count_nonzero(a != np.rot90(a, k))) for k in (1, 2, 3))

def choose_best_subset(bits_list, target_count):
    n = len(bits_list)

    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = rot_hamming(bits_list[i], bits_list[j])
            dist[i, j] = dist[j, i] = d

    self_dist = [self_rot_distance(b) for b in bits_list]

    best_threshold = None
    best_clique = None

    for threshold in range(bits_list[0].size, 0, -1):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(
            (i, j) for i in range(n) for j in range(i + 1, n)
            if dist[i, j] >= threshold
        )

        clique = max(nx.find_cliques(G), key=len, default=[])
        if len(clique) >= target_count:
            best_threshold = threshold
            best_clique = sorted(clique)
            break

    if best_threshold is None:
        raise RuntimeError(f"Could not find a {target_count}-marker subset.")

    # If the clique is small enough, search all target_count subsets exactly
    # and maximize:
    #   1) minimum pairwise rotational distance
    #   2) average pairwise rotational distance
    #   3) sum of self-rotation distances
    if len(best_clique) <= 28:
        best_score = None
        best_subset = None

        for subset in itertools.combinations(best_clique, target_count):
            ds = [
                int(dist[i, j])
                for idx, i in enumerate(subset)
                for j in subset[idx + 1:]
            ]
            score = (
                min(ds),
                sum(ds) / len(ds),
                sum(self_dist[i] for i in subset),
                tuple(subset)
            )
            if best_score is None or score > best_score:
                best_score = score
                best_subset = list(subset)
    else:
        # Greedy fallback
        remaining = set(best_clique)
        best_subset = []
        while len(best_subset) < target_count:
            if not best_subset:
                candidate = max(remaining, key=lambda i: self_dist[i])
            else:
                candidate = max(
                    remaining,
                    key=lambda i: (
                        min(dist[i, j] for j in best_subset),
                        sum(dist[i, j] for j in best_subset),
                        self_dist[i],
                    )
                )
            best_subset.append(candidate)
            remaining.remove(candidate)

        best_score = None
        ds = [
            int(dist[i, j])
            for idx, i in enumerate(best_subset)
            for j in best_subset[idx + 1:]
        ]
        best_score = (min(ds), sum(ds) / len(ds), sum(self_dist[i] for i in best_subset), tuple(best_subset))

    return best_subset, dist, self_dist, best_threshold, best_score

def write_txt(path, bits_list, source_ids, dist):
    count = len(source_ids)

    with open(path, "w", encoding="utf-8") as f:
        f.write("marker_size = 4x4\n")
        f.write(f"num_markers = {count}\n")

        # exact minimum pairwise rotational distance inside chosen subset
        pairwise = []
        for a in range(count):
            for b in range(a + 1, count):
                d = int(dist[source_ids[a], source_ids[b]])
                pairwise.append(((a, b), d))
        best_min = min(d for _, d in pairwise)

        f.write(f"best_min_rotational_hamming_distance = {best_min}\n\n")
        f.write("Source dictionary = DICT_4X4_50\n")
        f.write("Bit convention = 1 means white, 0 means black\n")
        f.write("Payload only = inner 4x4 bits; renderer should add a 1-cell black border\n\n")

        f.write("Selected source marker IDs from DICT_4X4_50:\n")
        for new_id, src_id in enumerate(source_ids):
            f.write(f"  custom ID {new_id} <- source ID {src_id}\n")

        f.write("\nPairwise rotational distances:\n")
        for (a, b), d in pairwise:
            f.write(f"  ID {a} vs ID {b}: {d}\n")

        f.write("\nMarker bit patterns:\n\n")
        for new_id, src_id in enumerate(source_ids):
            bits = bits_list[src_id]
            f.write(f"ID {new_id}:\n")
            for row in bits:
                f.write(" ".join(str(int(x)) for x in row) + "\n")
            f.write("\n")

def main():
    d = cv2.aruco.getPredefinedDictionary(SOURCE_DICT)
    bits_list = [
        d.getBitsFromByteList(d.bytesList[i:i+1], d.markerSize)
        for i in range(d.bytesList.shape[0])
    ]

    subset, dist, self_dist, threshold, score = choose_best_subset(bits_list, TARGET_COUNT)

    print("Selected source IDs from DICT_4X4_50:")
    print(subset)
    print(f"Maximum achievable minimum rotational distance for {TARGET_COUNT} markers: {threshold}")
    print(f"Chosen subset score (min_dist, avg_dist, sum_self_rot_dist, ids): {score}")

    write_txt(OUT_TXT, bits_list, subset, dist)
    print(f"Wrote {OUT_TXT}")

if __name__ == "__main__":
    main()
