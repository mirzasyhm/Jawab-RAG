
JSONL_FILE_PATH = 'quran_data_rag.jsonl'

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

QWEN_MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LEN_QWEN = 8192

TOP_K_BM25 = 5
TOP_K_EMBEDDING = 5
JOIN_TOP_K = 7

# Surah name to number mapping
SURAH_NAME_TO_NO = {
    "al-fatihah": 1, "al-baqarah": 2, "ali-imran": 3, "an-nisa": 4,
    "al-maidah": 5, "al-anam": 6, "al-araf": 7, "al-anfal": 8,
    "at-tawbah": 9, "yunus": 10, "hud": 11, "yusuf": 12,
    "ar-rad": 13, "ibrahim": 14, "al-hijr": 15, "an-nahl": 16,
    "al-isra": 17, "al-kahf": 18, "maryam": 19, "ta-ha": 20,
    "al-anbiya": 21, "al-hajj": 22, "al-mum'inun": 23, "an-nur": 24, # Note: Original had "al-mum'inun", standard is "al-muminun"
    "al-furqan": 25, "ash-shuara": 26, "an-naml": 27, "al-qasas": 28,
    "al-ankabut": 29, "ar-rum": 30, "luqman": 31, "as-sajdah": 32,
    "al-ahzab": 33, "saba": 34, "fatir": 35, "ya-sin": 36,
    "as-saffat": 37, "sad": 38, "az-zumar": 39, "ghafir": 40,
    "fussilat": 41, "ash-shura": 42, "az-zukhruf": 43, "ad-dukhan": 44,
    "al-jathiyah": 45, "al-ahqaf": 46, "muhammad": 47, "al-fath": 48,
    "al-hujurat": 49, "qaf": 50, "adh-dhariyat": 51, "at-tur": 52,
    "an-najm": 53, "al-qamar": 54, "ar-rahman": 55, "al-waqiah": 56,
    "al-hadid": 57, "al-mujadila": 58, "al-hashr": 59, "al-mumtahanah": 60,
    "as-saff": 61, "al-jumuah": 62, "al-munafiqun": 63, "at-taghabun": 64,
    "at-talaq": 65, "at-tahrim": 66, "al-mulk": 67, "al-qalam": 68,
    "al-haqqah": 69, "al-maarij": 70, "nuh": 71, "al-jinn": 72,
    "al-muzzammil": 73, "al-muddaththir": 74, "al-qiyamah": 75, "al-insan": 76,
    "al-mursalat": 77, "an-naba": 78, "an-naziat": 79, "abasa": 80,
    "at-takwir": 81, "al-infitar": 82, "al-mutaffifin": 83, "al-inshiqaq": 84,
    "al-buruj": 85, "at-tariq": 86, "al-aala": 87, "al-ghashiyah": 88,
    "al-fajr": 89, "al-balad": 90, "ash-shams": 91, "al-layl": 92,
    "ad-duha": 93, "ash-sharh": 94, "at-tin": 95, "al-alaq": 96,
    "al-qadr": 97, "al-bayyinah": 98, "az-zalzalah": 99, "al-adiyat": 100,
    "al-qariah": 101, "at-takathur": 102, "al-asr": 103, "al-humazah": 104,
    "al-fil": 105, "quraysh": 106, "al-maun": 107, "al-kawthar": 108,
    "al-kafirun": 109, "an-nasr": 110, "al-masad": 111, "al-ikhlas": 112,
    "al-falaq": 113, "an-nas": 114
}

# Reverse mapping for Surah number to name
SURAH_NO_TO_NAME = {v: k for k, v in SURAH_NAME_TO_NO.items()}
