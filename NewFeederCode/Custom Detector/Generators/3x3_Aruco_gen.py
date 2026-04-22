import os
import itertools
import numpy as np
import cv2

# =========================
# Configuration
# =========================
MARKER_SIZE = 4 # 3x3 payload
NUM_MARKERS = 20 # "3x3_5"
OUT_DIR = "aruco_4x4_20"
MARKER_PIXELS = 300 # output image size for each marker
BORDER_BITS = 1 # standard black border width in marker-bit units

# =========================
# Bit / rotation utilities
# =========================
def int_to_bits(n, size=3):
    """
    Convert integer -> size x size binary matrix (row-major).
    """
    bits = [(n >> (size * size - 1 - i)) & 1 for i in range(size * size)]
    return np.array(bits, dtype=np.uint8).reshape(size, size)

def rotate_bits(bits, k):
    """
    Rotate marker by k * 90 degrees CCW.
    """
    return np.rot90(bits, k)

def all_rotations(bits):
    return [rotate_bits(bits, k) for k in range(4)]

def is_rotation_unique(bits):
    """
    True if all 4 rotations are different.
    """
    reps = {tuple(r.flatten().tolist()) for r in all_rotations(bits)}
    return len(reps) == 4

def canonical_rotation(bits):
    """
    Pick one canonical representative for a rotation family.
    We use the lexicographically smallest flattened rotation.
    """
    rots = all_rotations(bits)
    flat_rots = [tuple(r.flatten().tolist()) for r in rots]
    best_idx = min(range(4), key=lambda i: flat_rots[i])
    return rots[best_idx]

def rotational_distance(bits_a, bits_b):
    """
    Distance between two marker families:
    minimum Hamming distance over all relative rotations.
    """
    best = MARKER_SIZE * MARKER_SIZE + 1
    for ra in all_rotations(bits_a):
        for rb in all_rotations(bits_b):
            dist = int(np.count_nonzero(ra != rb))
            if dist < best:
                best = dist
    return best

# =========================
# Candidate generation
# =========================
def generate_rotation_unique_families(size=3):
    """
    Enumerate all 2^(size*size) markers, keep only those that are
    orientation-resolvable, then deduplicate by rotation.
    """
    families = {}
    for n in range(2 ** (size * size)):
        bits = int_to_bits(n, size)
        if not is_rotation_unique(bits):
            continue
        canon = canonical_rotation(bits)
        key = tuple(canon.flatten().tolist())
        families[key] = canon

    candidates = list(families.values())
    return candidates

# =========================
# Exact search:
# maximize minimum pairwise rotational distance
# =========================
def build_distance_matrix(candidates):
    n = len(candidates)
    D = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            d = rotational_distance(candidates[i], candidates[j])
            D[i, j] = d
            D[j, i] = d
    return D

def find_any_k_clique(adj, k):
    """
    Return any clique of size k in graph given by adjacency bitsets.
    Exact recursive search.
    """
    n = len(adj)
    all_nodes = (1 << n) - 1
    
    def pop_lsb(x):
        lsb = x & -x
        idx = lsb.bit_length() - 1
        return idx, x ^ lsb

    found = None

    def dfs(chosen_bits, candidates_bits):
        nonlocal found
        if found is not None:
            return

        chosen_count = bin(chosen_bits).count('1')
        cand_count = bin(candidates_bits).count('1')

        if chosen_count == k:
            found = chosen_bits
            return
        if chosen_count + cand_count < k:
            return

        temp = candidates_bits
        while temp and found is None:
            v, temp = pop_lsb(temp)
            dfs(chosen_bits | (1 << v), candidates_bits & adj[v])
            candidates_bits &= ~(1 << v)
            if bin(chosen_bits).count('1') + bin(candidates_bits).count('1') < k:
                return

    dfs(0, all_nodes)
    if found is None:
        return None

    return [i for i in range(n) if (found >> i) & 1]

def best_dictionary_indices(candidates, k):
    """
    Exact max-min search.
    """
    D = build_distance_matrix(candidates)
    n = len(candidates)

    max_possible = MARKER_SIZE * MARKER_SIZE
    best_t = None
    best_clique = None

    for t in range(max_possible, 0, -1):
        adj = [0] * n
        for i in range(n):
            bits = 0
            for j in range(n):
                if i != j and D[i, j] >= t:
                    bits |= (1 << j)
            adj[i] = bits

        clique = find_any_k_clique(adj, k)
        if clique is not None:
            best_t = t
            best_clique = clique
            break

    if best_clique is None:
        raise RuntimeError("Could not find a valid set of markers.")

    return best_clique, best_t, D

# =========================
# OpenCV custom dictionary creation
# =========================
def get_byte_list_from_bits(bits):
    """
    Works across OpenCV versions.
    """
    if hasattr(cv2.aruco.Dictionary, "getByteListFromBits"):
        return cv2.aruco.Dictionary.getByteListFromBits(bits)
    return cv2.aruco.Dictionary_getByteListFromBits(bits)

def make_custom_dictionary(selected_markers, min_distance):
    """
    Build cv2.aruco.Dictionary from selected bit matrices.
    """
    rows = [get_byte_list_from_bits(bits) for bits in selected_markers]
    bytes_list = np.concatenate(rows, axis=0)
    maxcorr = max(0, (min_distance - 1) // 2)

    aruco_dict = cv2.aruco.Dictionary(bytes_list, MARKER_SIZE, maxcorr)
    return aruco_dict

# =========================
# Save outputs
# =========================
def save_marker_images(aruco_dict, out_dir, side_pixels=300, border_bits=1):
    os.makedirs(out_dir, exist_ok=True)
    for marker_id in range(len(aruco_dict.bytesList)):
        img = cv2.aruco.generateImageMarker(
            aruco_dict,
            marker_id,
            side_pixels,
            borderBits=border_bits
        )
        cv2.imwrite(os.path.join(out_dir, f"marker_{marker_id}.png"), img)

def save_contact_sheet(aruco_dict, out_dir, side_pixels=200, border_bits=1):
    margin = 20
    label_h = 40
    cols = NUM_MARKERS
    rows = 1
    W = cols * side_pixels + (cols + 1) * margin
    H = rows * (side_pixels + label_h) + (rows + 1) * margin
    canvas = np.ones((H, W), dtype=np.uint8) * 255

    for i in range(NUM_MARKERS):
        img = cv2.aruco.generateImageMarker(
            aruco_dict,
            i,
            side_pixels,
            borderBits=border_bits
        )

        x = margin + i * (side_pixels + margin)
        y = margin
        canvas[y:y + side_pixels, x:x + side_pixels] = img

        cv2.putText(
            canvas,
            f"ID {i}",
            (x + 20, y + side_pixels + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            0,
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(os.path.join(out_dir, "dictionary_overview.png"), canvas)

def save_selected_bits_txt(selected_markers, out_dir, min_distance, D, indices):
    path = os.path.join(out_dir, "selected_markers.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"marker_size = {MARKER_SIZE}x{MARKER_SIZE}\n")
        f.write(f"num_markers = {len(selected_markers)}\n")
        f.write(f"best_min_rotational_hamming_distance = {min_distance}\n\n")
        f.write("Pairwise rotational distances:\n")
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                f.write(f"  ID {i} vs ID {j}: {D[indices[i], indices[j]]}\n")

        f.write("\nMarker bit patterns:\n")
        for i, bits in enumerate(selected_markers):
            f.write(f"\nID {i}:\n")
            for row in bits:
                f.write(" ".join(str(int(v)) for v in row) + "\n")

# =========================
# Main
# =========================
def main():
    candidates = generate_rotation_unique_families(MARKER_SIZE)
    print(f"Rotation-unique 3x3 marker families found: {len(candidates)}")
    
    indices, best_min_dist, D = best_dictionary_indices(candidates, NUM_MARKERS)
    selected_markers = [candidates[i] for i in indices]

    print(f"Best minimum rotational Hamming distance for {NUM_MARKERS} markers: {best_min_dist}")
    print("\nSelected markers:\n")
    for i, bits in enumerate(selected_markers):
        print(f"ID {i}:")
        print(bits)
        print()

    aruco_dict = make_custom_dictionary(selected_markers, best_min_dist)

    save_marker_images(aruco_dict, OUT_DIR, MARKER_PIXELS, BORDER_BITS)
    save_contact_sheet(aruco_dict, OUT_DIR, 200, BORDER_BITS)
    save_selected_bits_txt(selected_markers, OUT_DIR, best_min_dist, D, indices)

    print(f"Saved dictionary markers to: {OUT_DIR}")

if __name__ == "__main__":
    main()