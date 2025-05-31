# HAWP ile Tenis Kortu Tespiti (CPU için Düzenlenmiş)

Bu depo, CPU üzerinde çalışacak şekilde yapılandırılmış, düzenlenmiş bir HAWP modeli kullanarak resimlerdeki tenis kortu çizgilerini tespit etmek için kod içerir.

## Kurulum ve Yükleme

1.  **Depoyu Klonlayın:**

    ```bash
    git clone https://github.com/islekeren/tennis-court-detection
    cd tennis-court-detection
    git submodule init
    git submodule update --remote
    ```

2.  **Python Ortamı Oluşturun:**
    Sanal bir ortam kullanmanız önerilir.

    - venv kullanarak:
      ```bash
      python -m venv venv
      # Windows'ta
      .\venv\Scripts\activate
      # macOS/Linux'ta
      source venv/bin/activate
      ```

3.  **PyTorch'u Kurun (CPU versiyonu):**

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

    Kurulumu doğrulayın:

    ```bash
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
    # İkinci print çıktısı False olmalıdır
    ```

4.  **Genel Bağımlılıkları Kurun:**

    ```bash
    pip install scikit-learn numpy matplotlib Pillow docopt opencv-python networkx shapely tqdm
    pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    ```

5.  **HAWP Alt Modül Bağımlılıklarını Kurun:**
    HAWP dizinine gidin ve gereksinimlerini kurun.

    ```bash
    cd hawp
    pip install -e .
    # -e . komutu, hawp paketini mevcut dizinden düzenlenebilir modda kurar.
    # Kapsanmamışsa özel gereksinimlerini de kurmanız gerekebilir:
    # pip install -r requirement.txt
    cd ..
    # Proje kök dizinine geri dönün
    ```

6.  **HAWP Önceden Eğitilmiş Modellerini İndirin:**
    - `hawp` dizinine gidin: `cd hawp`
    - Eğer yoksa bir `checkpoints` dizini oluşturun: `mkdir checkpoints`
    - HAWP model ağırlıklarını indirin. `hawpv2-edb9b23f.pth` modeli varsayılan yapılandırma tarafından kullanılır.
      Veya `curl` kullanarak (macOS/Linux'ta ve genellikle Windows'ta mevcuttur):
      ```bash
      curl -L https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv2/hawpv2-edb9b23f.pth -o checkpoints/hawpv2-edb9b23f.pth
      ```
    - Proje kök dizinine geri dönün: `cd ..`

## Tespiti Çalıştırma

1.  **Çıktı Dizinini Kontrol Edin:**
    Betik, logları `output/ihawp/` dizinine kaydetmeye çalışacaktır. Eğer yoksa bu dizini oluşturun:

    ```bash
    mkdir -p output/ihawp
    # -p bayrağı, gerektiğinde üst dizinleri oluşturur ve zaten varsa hata vermez.
    # Windows Komut İstemi'nde şuna ihtiyacınız olabilir: mkdir output\ihawp ('output' yoksa önce onu oluşturun)
    ```

    Alternatif olarak, bu depodaki `modelFitting.py` betiği bu dizini otomatik olarak oluşturacak şekilde düzenlenmiştir.

2.  **`modelFitting.py` Betiğini Çalıştırın:**
    Betiği projenin kök dizininden çalıştırın.
    ```bash
    python modelFitting.py --config-file hawp/configs/hawpv2.yaml --img path/to/your/image.png --output_path path/to/your/output_image.png --threshold 0.8
    ```
    - `path/to/your/image.png` kısmını işlemek istediğiniz resmin yoluyla değiştirin.
    - `path/to/your/output_image.png` kısmını sonucu kaydetmek istediğiniz yerle değiştirin.
    - `--threshold` değerini (örneğin, `0.5` ila `0.97`) gerektiği gibi ayarlayın. Daha düşük bir eşik değeri daha fazla çizgi tespit edebilir ancak daha fazla yanlış pozitif sonuç da içerebilir.
