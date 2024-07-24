import sys, os

# Aktuellen Dateipfad ermitteln
current_file_path = os.path.abspath(__name__)

# Übergeordneten Ordner ermitteln
parent_directory = os.path.dirname(os.path.dirname(current_file_path))

# Pfad zum 'src' Unterordner hinzufügen
src_path = os.path.join(parent_directory, "src")
sys.path.append(src_path)

from utils import clean


def test_clean():
    text = "This is a sample text with some stopwords and punctuation!"
    expected_result = "sampl text stopword punctuat"
    assert clean(text) == expected_result
