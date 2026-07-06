"""
=====================================================================
 TUGAS REKAYASA KOMPUTASIONAL
 Implementasi Algoritma Genetika (Genetic Algorithm / GA)
 untuk Pencarian Kata dalam Kamus Bahasa Daerah (Bahasa Bugis - Dialek Bone)
=====================================================================

Konsep GA yang dipakai (mengikuti modul perkuliahan):
  - Individu   : deretan huruf (kromosom) sepanjang kata target
  - Gen        : satu huruf pada posisi tertentu dalam individu
  - Fitness    : jumlah huruf yang posisinya cocok dengan kata target,
                 dibagi panjang kata target (nilai 0 - 1)
  - Seleksi    : Roulette Wheel Selection (berbasis probabilitas kumulatif)
  - Crossover  : Single Point Crossover
  - Mutasi     : Random Reset Mutation (huruf diganti huruf acak lain)

Alur satu generasi (menu 3) persis mengikuti contoh di modul:
  populasi awal -> hitung fitness -> seleksi roulette -> crossover
  -> mutasi -> populasi baru (generasi berikutnya)
"""

import random
import string

ALPHABET = string.ascii_uppercase  # A - Z, dipakai untuk membangun individu acak


# ---------------------------------------------------------------------------
# 1. DATABASE KAMUS BAHASA DAERAH (Bugis - Dialek Bone) - minimal 10 data kata
# ---------------------------------------------------------------------------
KAMUS = [
    {"kata": "BOLA",   "arti": "rumah"},
    {"kata": "NANRE",  "arti": "nasi"},
    {"kata": "WAE",    "arti": "air"},
    {"kata": "GALUNG", "arti": "sawah"},
    {"kata": "LOPI",   "arti": "perahu"},
    {"kata": "JAMA",   "arti": "kerja"},
    {"kata": "TINRO",  "arti": "tidur"},
    {"kata": "CINNA",  "arti": "ingin / suka"},
    {"kata": "TANAE",  "arti": "tanah"},
    {"kata": "MABELA", "arti": "jauh"},
    {"kata": "BOSI",   "arti": "hujan"},
    {"kata": "MACCA",  "arti": "pintar"},
]


# ---------------------------------------------------------------------------
# 2. STATE / PARAMETER ALGORITMA GENETIKA
# ---------------------------------------------------------------------------
class GAState:
    """Menyimpan seluruh kondisi & hasil perhitungan GA agar bisa
    ditampilkan kembali lewat menu 4-9 tanpa perlu menghitung ulang."""

    def __init__(self):
        self.target_word = None        # kata target yang sedang dicari
        self.target_meaning = None
        self.population = []           # populasi AKTIF saat ini (generasi terbaru)
        self.pop_size = 6
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.generation_number = 0
        self.found = False
        self.last_run = None           # dict berisi rincian generasi yang BARU dihitung

    def reset_run(self):
        self.target_word = None
        self.target_meaning = None
        self.population = []
        self.generation_number = 0
        self.found = False
        self.last_run = None


STATE = GAState()


# ---------------------------------------------------------------------------
# 3. FUNGSI-FUNGSI INTI ALGORITMA GENETIKA
# ---------------------------------------------------------------------------
def buat_individu_acak(panjang):
    """Membuat satu individu (kromosom) acak sepanjang `panjang` huruf."""
    return "".join(random.choice(ALPHABET) for _ in range(panjang))


def buat_populasi_awal(panjang, jumlah):
    return [buat_individu_acak(panjang) for _ in range(jumlah)]


def hitung_fitness(individu, target):
    """Fitness = (jumlah huruf yang posisinya sama dengan target) / panjang target."""
    cocok = sum(1 for a, b in zip(individu, target) if a == b)
    return cocok, cocok / len(target)


def hitung_semua_fitness(populasi, target):
    """Mengembalikan list of dict: individu, huruf_cocok, fitness."""
    hasil = []
    for ind in populasi:
        cocok, fit = hitung_fitness(ind, target)
        hasil.append({"individu": ind, "huruf_cocok": cocok, "fitness": fit})
    return hasil


def hitung_probabilitas_interval(fitness_list):
    """Menghitung probabilitas dan interval kumulatif untuk roulette wheel."""
    total_fitness = sum(f["fitness"] for f in fitness_list)
    hasil = []
    batas_bawah = 0.0
    for f in fitness_list:
        if total_fitness == 0:
            # Jika seluruh fitness 0, beri probabilitas sama rata agar
            # roda roulette tetap bisa berputar (tidak pernah 0/0)
            prob = 1 / len(fitness_list)
        else:
            prob = f["fitness"] / total_fitness
        batas_atas = batas_bawah + prob
        hasil.append({
            "individu": f["individu"],
            "fitness": f["fitness"],
            "probabilitas": prob,
            "interval_bawah": batas_bawah,
            "interval_atas": batas_atas,
        })
        batas_bawah = batas_atas
    return hasil


def seleksi_roulette(tabel_interval, jumlah_dipilih):
    """Memutar 'roda roulette' sebanyak jumlah_dipilih kali.
    Mengembalikan list dict {r, individu_terpilih}."""
    terpilih = []
    for _ in range(jumlah_dipilih):
        r = random.random()  # angka acak [0.0, 1.0)
        for baris in tabel_interval:
            if baris["interval_bawah"] <= r < baris["interval_atas"]:
                terpilih.append({"r": r, "individu": baris["individu"]})
                break
        else:
            # fallback pembulatan (r == 1.0 dsb) -> ambil individu terakhir
            terpilih.append({"r": r, "individu": tabel_interval[-1]["individu"]})
    return terpilih


def crossover_satu_titik(induk1, induk2, crossover_rate):
    """Single Point Crossover. Mengembalikan (anak1, anak2, titik_potong, dilakukan)."""
    panjang = len(induk1)
    if panjang < 2 or random.random() > crossover_rate:
        # tidak crossover -> anak = salinan induk
        return induk1, induk2, None, False
    titik = random.randint(1, panjang - 1)  # posisi potong, minimal 1 huruf tersisa
    anak1 = induk1[:titik] + induk2[titik:]
    anak2 = induk2[:titik] + induk1[titik:]
    return anak1, anak2, titik, True


def mutasi(individu, mutation_rate):
    """Random reset mutation per gen (huruf). Mengembalikan (hasil, list_posisi_mutasi)."""
    huruf = list(individu)
    posisi_mutasi = []
    for i in range(len(huruf)):
        if random.random() < mutation_rate:
            huruf_baru = random.choice(ALPHABET.replace(huruf[i], "") or ALPHABET)
            huruf[i] = huruf_baru
            posisi_mutasi.append(i + 1)  # posisi 1-based, biar sesuai gaya modul
    return "".join(huruf), posisi_mutasi


# ---------------------------------------------------------------------------
# 4. FUNGSI UTAMA MENJALANKAN 1 GENERASI (dipanggil oleh menu 3)
# ---------------------------------------------------------------------------
def jalankan_satu_generasi():
    target = STATE.target_word
    populasi_awal = STATE.population

    # --- Langkah 1: Hitung fitness populasi saat ini ---
    fitness_list = hitung_semua_fitness(populasi_awal, target)

    # --- Langkah 2: Seleksi Roulette Wheel ---
    tabel_interval = hitung_probabilitas_interval(fitness_list)
    mating_pool = seleksi_roulette(tabel_interval, STATE.pop_size)

    # --- Langkah 3: Crossover (berpasangan berurutan dari mating pool) ---
    hasil_crossover = []
    individu_pool = [m["individu"] for m in mating_pool]
    for i in range(0, len(individu_pool) - 1, 2):
        p1, p2 = individu_pool[i], individu_pool[i + 1]
        a1, a2, titik, dilakukan = crossover_satu_titik(p1, p2, STATE.crossover_rate)
        hasil_crossover.append({
            "induk1": p1, "induk2": p2,
            "titik_potong": titik, "dilakukan": dilakukan,
            "anak1": a1, "anak2": a2,
        })
    if len(individu_pool) % 2 == 1:
        # individu ganjil terakhir dibawa langsung tanpa pasangan
        sisa = individu_pool[-1]
        hasil_crossover.append({
            "induk1": sisa, "induk2": None,
            "titik_potong": None, "dilakukan": False,
            "anak1": sisa, "anak2": None,
        })

    anak_semua = []
    for hc in hasil_crossover:
        anak_semua.append(hc["anak1"])
        if hc["anak2"] is not None:
            anak_semua.append(hc["anak2"])

    # --- Langkah 4: Mutasi setiap anak hasil crossover ---
    hasil_mutasi = []
    populasi_baru = []
    for anak in anak_semua:
        hasil, posisi = mutasi(anak, STATE.mutation_rate)
        hasil_mutasi.append({"sebelum": anak, "sesudah": hasil, "posisi_mutasi": posisi})
        populasi_baru.append(hasil)

    # --- Langkah 5: Elitisme ringan -> individu terbaik generasi lama tetap
    #     dipertahankan agar solusi terbaik tidak pernah hilang (opsional,
    #     praktik umum GA supaya hasil konsisten membaik tiap generasi) ---
    terbaik_lama = max(fitness_list, key=lambda x: x["fitness"])
    fitness_baru = hitung_semua_fitness(populasi_baru, target)
    terburuk_baru_idx = min(range(len(fitness_baru)), key=lambda i: fitness_baru[i]["fitness"])
    if terbaik_lama["fitness"] > fitness_baru[terburuk_baru_idx]["fitness"]:
        populasi_baru[terburuk_baru_idx] = terbaik_lama["individu"]

    fitness_populasi_baru = hitung_semua_fitness(populasi_baru, target)
    terbaik_baru = max(fitness_populasi_baru, key=lambda x: x["fitness"])

    STATE.generation_number += 1
    STATE.population = populasi_baru
    STATE.found = terbaik_baru["individu"] == target

    STATE.last_run = {
        "generasi_ke": STATE.generation_number,
        "populasi_awal": populasi_awal,
        "fitness_awal": fitness_list,
        "tabel_interval": tabel_interval,
        "mating_pool": mating_pool,
        "hasil_crossover": hasil_crossover,
        "hasil_mutasi": hasil_mutasi,
        "populasi_baru": populasi_baru,
        "fitness_baru": fitness_populasi_baru,
        "terbaik_baru": terbaik_baru,
    }
    return STATE.last_run


# ---------------------------------------------------------------------------
# 5. FUNGSI-FUNGSI TAMPILAN MENU
# ---------------------------------------------------------------------------
def cetak_judul(judul):
    print("\n" + "=" * 60)
    print(judul)
    print("=" * 60)


def menu_1_tampilkan_kamus():
    cetak_judul("1. TAMPILKAN KAMUS BAHASA DAERAH")
    print(f"{'No':<4}{'Kata':<12}{'Arti':<20}")
    print("-" * 36)
    for i, entri in enumerate(KAMUS, start=1):
        print(f"{i:<4}{entri['kata']:<12}{entri['arti']:<20}")


def menu_2_cari_kata():
    cetak_judul("2. CARI KATA")
    kunci = input("Masukkan kata / arti yang dicari: ").strip().lower()
    hasil = [e for e in KAMUS if kunci in e["kata"].lower() or kunci in e["arti"].lower()]
    if hasil:
        print(f"\nDitemukan {len(hasil)} hasil:")
        for e in hasil:
            print(f"  {e['kata']}  ->  {e['arti']}")
    else:
        print("Kata/arti tidak ditemukan dalam kamus.")


def pilih_kata_target():
    menu_1_tampilkan_kamus()
    print()
    while True:
        pilihan = input(f"Pilih nomor kata target (1-{len(KAMUS)}): ").strip()
        if pilihan.isdigit() and 1 <= int(pilihan) <= len(KAMUS):
            entri = KAMUS[int(pilihan) - 1]
            return entri["kata"].upper(), entri["arti"]
        print("Input tidak valid, coba lagi.")


def menu_3_jalankan_ga():
    cetak_judul("3. JALANKAN ALGORITMA GENETIKA")

    if STATE.target_word is None:
        # ----- Inisialisasi run GA yang baru -----
        target, arti = pilih_kata_target()
        try:
            ukuran = int(input(f"Ukuran populasi (disarankan genap, contoh 6) [default 6]: ") or 6)
        except ValueError:
            ukuran = 6
        STATE.reset_run()
        STATE.target_word = target
        STATE.target_meaning = arti
        STATE.pop_size = max(2, ukuran)
        STATE.population = buat_populasi_awal(len(target), STATE.pop_size)
        print(f"\nKata target dipilih : {target}  (artinya: {arti})")
        print(f"Populasi awal dibuat secara acak sejumlah {STATE.pop_size} individu.")
    else:
        print(f"Melanjutkan pencarian kata target: {STATE.target_word} "
              f"(sudah generasi ke-{STATE.generation_number})")

    hasil = jalankan_satu_generasi()

    print(f"\n>> Generasi ke-{hasil['generasi_ke']} selesai dihitung.")
    print(f">> Individu terbaik generasi ini : {hasil['terbaik_baru']['individu']} "
          f"(fitness = {hasil['terbaik_baru']['fitness']:.2f})")
    if STATE.found:
        print(f">> TARGET '{STATE.target_word}' BERHASIL DITEMUKAN!")
    print("\nSilakan buka menu 4-9 untuk melihat rincian tiap tahap "
          "(populasi, fitness, roulette, crossover, mutasi, generasi baru).")


def _pastikan_sudah_jalan():
    if STATE.last_run is None:
        print("\nBelum ada proses GA yang dijalankan. Silakan pilih menu 3 terlebih dahulu.")
        return False
    return True


def menu_4_tampilkan_populasi():
    cetak_judul("4. TAMPILKAN POPULASI")
    if not _pastikan_sudah_jalan():
        return
    lr = STATE.last_run
    print(f"Kata target        : {STATE.target_word}")
    print(f"Generasi ke         : {lr['generasi_ke']}")
    print(f"\nPopulasi AWAL (sebelum proses generasi ke-{lr['generasi_ke']}):")
    for i, ind in enumerate(lr["populasi_awal"], start=1):
        print(f"  I{i}: {ind}")


def menu_5_hasil_fitness():
    cetak_judul("5. HASIL FITNESS")
    if not _pastikan_sudah_jalan():
        return
    lr = STATE.last_run
    print(f"Kata target: {STATE.target_word}  (panjang {len(STATE.target_word)} huruf)\n")
    print(f"{'Individu':<10}{'Kromosom':<14}{'Huruf Cocok':<14}{'Fitness':<10}")
    print("-" * 48)
    for i, f in enumerate(lr["fitness_awal"], start=1):
        print(f"{'I' + str(i):<10}{f['individu']:<14}{f['huruf_cocok']:<14}{f['fitness']:<10.2f}")


def menu_6_seleksi_roulette():
    cetak_judul("6. SELEKSI ROULETTE")
    if not _pastikan_sudah_jalan():
        return
    lr = STATE.last_run
    print("Tabel Probabilitas dan Interval:")
    print(f"{'Individu':<10}{'Fitness':<10}{'Probabilitas':<14}{'Interval'}")
    print("-" * 55)
    for i, b in enumerate(lr["tabel_interval"], start=1):
        interval_str = f"{b['interval_bawah']:.2f} - {b['interval_atas']:.2f}"
        print(f"{'I' + str(i):<10}{b['fitness']:<10.2f}{b['probabilitas']:<14.2f}{interval_str}")

    print("\nProses Pemutaran Roda Roulette (angka acak r dibandingkan ke interval):")
    print(f"{'Putaran':<10}{'r':<10}{'Individu Terpilih'}")
    print("-" * 35)
    for i, m in enumerate(lr["mating_pool"], start=1):
        print(f"{i:<10}{m['r']:<10.3f}{m['individu']}")


def menu_7_crossover():
    cetak_judul("7. CROSS OVER")
    if not _pastikan_sudah_jalan():
        return
    lr = STATE.last_run
    for i, hc in enumerate(lr["hasil_crossover"], start=1):
        print(f"Pasangan {i}:")
        print(f"  Induk 1        : {hc['induk1']}")
        if hc["induk2"] is not None:
            print(f"  Induk 2        : {hc['induk2']}")
            if hc["dilakukan"]:
                print(f"  Titik potong   : setelah huruf ke-{hc['titik_potong']}")
                print(f"  Anak 1         : {hc['anak1']}")
                print(f"  Anak 2         : {hc['anak2']}")
            else:
                print("  (Tidak terjadi crossover, anak = salinan induk)")
                print(f"  Anak 1         : {hc['anak1']}")
                print(f"  Anak 2         : {hc['anak2']}")
        else:
            print("  (Individu tanpa pasangan, dibawa langsung ke generasi berikut)")
            print(f"  Anak           : {hc['anak1']}")
        print()


def menu_8_mutasi():
    cetak_judul("8. MUTASI")
    if not _pastikan_sudah_jalan():
        return
    lr = STATE.last_run
    print(f"Peluang mutasi per gen: {STATE.mutation_rate:.2f}\n")
    for i, hm in enumerate(lr["hasil_mutasi"], start=1):
        posisi = ", ".join(str(p) for p in hm["posisi_mutasi"]) or "-"
        print(f"Anak {i}:")
        print(f"  Sebelum mutasi : {hm['sebelum']}")
        print(f"  Sesudah mutasi : {hm['sesudah']}")
        print(f"  Posisi mutasi  : {posisi}")
        print()


def menu_9_generasi_baru():
    cetak_judul("9. GENERASI BARU")
    if not _pastikan_sudah_jalan():
        return
    lr = STATE.last_run
    print(f"Populasi setelah generasi ke-{lr['generasi_ke']}:\n")
    print(f"{'Individu':<10}{'Kromosom':<14}{'Fitness'}")
    print("-" * 34)
    for i, f in enumerate(lr["fitness_baru"], start=1):
        print(f"{'I' + str(i):<10}{f['individu']:<14}{f['fitness']:<.2f}")
    print(f"\nIndividu terbaik : {lr['terbaik_baru']['individu']} "
          f"(fitness = {lr['terbaik_baru']['fitness']:.2f})")
    if STATE.found:
        print(f"Kata target '{STATE.target_word}' sudah DITEMUKAN pada generasi ini.")
    else:
        print("Kata target belum ditemukan. Jalankan menu 3 lagi untuk melanjutkan "
              "ke generasi berikutnya.")


# ---------------------------------------------------------------------------
# 6. LOOP MENU UTAMA
# ---------------------------------------------------------------------------
def tampilkan_menu():
    print("\n" + "=" * 40)
    print("=== Kamus Bahasa Daerah (Bugis - Dialek Bone) ===")
    print("=" * 40)
    print("1. Tampilkan Kamus")
    print("2. Cari Kata")
    print("3. Jalankan Algoritma Genetika")
    print("4. Tampilkan Populasi")
    print("5. Hasil Fitness")
    print("6. Seleksi Roulette")
    print("7. Cross Over")
    print("8. Mutasi")
    print("9. Generasi Baru")
    print("10. Keluar")


def main():
    aksi = {
        "1": menu_1_tampilkan_kamus,
        "2": menu_2_cari_kata,
        "3": menu_3_jalankan_ga,
        "4": menu_4_tampilkan_populasi,
        "5": menu_5_hasil_fitness,
        "6": menu_6_seleksi_roulette,
        "7": menu_7_crossover,
        "8": menu_8_mutasi,
        "9": menu_9_generasi_baru,
    }
    while True:
        tampilkan_menu()
        pilihan = input("Pilih menu (1-10): ").strip()
        if pilihan == "10":
            print("Terima kasih. Program selesai.")
            break
        fungsi = aksi.get(pilihan)
        if fungsi:
            fungsi()
        else:
            print("Pilihan tidak valid, silakan coba lagi.")


if __name__ == "__main__":
    main()
