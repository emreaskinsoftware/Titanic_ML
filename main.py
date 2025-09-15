from src.data_loader import load_data, quick_audit

def main():
    # Veriyi yükle
    train, test = load_data()

    # Hızlı veri kontrolü yap
    quick_audit(train, test)

if __name__ == "__main__":
    main()
