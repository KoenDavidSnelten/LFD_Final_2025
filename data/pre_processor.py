import re
import html
# import emoji
import argparse
import os
import contractions

# def emoji_to_text(text):
#     """Convert emojis to text (e.g. 😀 → :grinning_face:)."""
#     return emoji.demojize(text, delimiters=(" ", " "))

def lower_case(text):
    """Convert text to lowercase."""
    return text.lower()

def remove_contractions(text):
    """Remove contractions."""
    expanded_words = []    
    for word in text.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))   
    return ' '.join(expanded_words).lower()

def remove_user(text):
    """Remove @USER mentions."""
    return re.sub(r"@USER", "", text)

def remove_url(text):
    """Remove actual URL"""
    return re.sub(r"URL", "", text)

def remove_hashtag(text):
    """Remove hashtags (#word → word)."""
    return re.sub(r"#(\w+)", r"\1", text)

def remove_html(text):
    """Decode HTML entities (&amp; → &, etc.)."""
    return html.unescape(text)

def open_file(filename):
    """Open dataset file with text and label separated by a tab."""
    with open(filename, encoding="utf-8") as f:
        lines = [line.strip().split("\t") for line in f if "\t" in line]
    return lines

def preprocess_pipeline(text, args):
    """Apply selected preprocessing steps."""
    if args.emoji_to_text:
        text = emoji_to_text(text)
    if args.lower_case:
        text = lower_case(text)
    if args.remove_user:
        text = remove_user(text)
    if args.remove_url:
        text = remove_url(text)
    if args.remove_hashtag:
        text = remove_hashtag(text)
    if args.remove_html:
        text = remove_html(text)
    if args.remove_contractions:
        text = remove_contractions(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    parser = argparse.ArgumentParser(description="Tweet preprocessing options")
    parser.add_argument("output_folder", help="Output folder")
    parser.add_argument("--emoji_to_text", action="store_true", help="Convert emojis to text")
    parser.add_argument("--lower_case", action="store_true", help="Convert text to lowercase")
    parser.add_argument("--remove_user", action="store_true", help="Remove @USER mentions")
    parser.add_argument("--remove_url", action="store_true", help="Remove URLs")
    parser.add_argument("--remove_hashtag", action="store_true", help="Remove hashtags")
    parser.add_argument("--remove_html", action="store_true", help="Decode HTML entities")
    parser.add_argument("--remove_contractions", action="store_true", help="remove_contractions")
    args = parser.parse_args()

    files = ["train.tsv", "dev.tsv", "test.tsv"]

    os.makedirs(args.output_folder, exist_ok=True)
    for file in files:
        input_path = "./raw/" + file
        output_path = os.path.join(args.output_folder, file)

        with open(input_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        with open(output_path, "w", encoding="utf-8") as f2:
            for line in lines:
                if "\t" in line:
                    text, label = line.split("\t", 1)
                    cleaned = preprocess_pipeline(text, args)
                    f2.write(f"{cleaned}\t{label}\n")
                else:
                    # If there’s no tab, just write the line back
                    f2.write(line + "\n")

        print(f"Processed {file} → {output_path}")



if __name__ == "__main__":
    main()
