from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import subprocess
import tempfile

app = Flask(__name__)

# Eğitimli Random Forest modelini yükle
model = joblib.load("random_forest_best_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ack")
def ack():
    return """
    <h1>Bu Araç Ne Yapar?</h1>
    <p>IoT cihazlarına gelen ağ trafiğini analiz ederek saldırı türünü tespit eder. 
    Tshark ile canlı trafik yakalanır ve makine öğrenmesi modeli ile sınıflandırılır.</p>
    """

@app.route("/visuals")
def visuals():
    # Görseller ve açıklamaları burada tanımlanır
    visuals = [
        {
            "filename": "AttackTypes1.png",
            "description": "İlk veri setinde yer alan saldırı türlerinin kategorik dağılımı görselleştirilmiştir."
        },
        {
            "filename": "PacketParameters2.png",
            "description": "İlk veri setindeki paket parametrelerinin sayısal analizi yapılmıştır."
        },
        {
            "filename": "AttackLabelVsAttackType3.png",
            "description": "İkinci veri setinde etiketler ile saldırı türleri arasındaki ilişki gösterilmiştir."
        },
        {
            "filename": "AttackDistrubution4.png",
            "description": "Birinci veri setinde saldırıların frekans dağılımı sunulmuştur."
        },
        {
            "filename": "FirstModelScore.jpg",
            "description": "İlk veri seti kullanılarak oluşturulan modelin doğruluk ve performans ölçümleri verilmiştir."
        },
        {
            "filename": "SecondModelScore.jpg",
            "description": "İkinci veri setiyle eğitilen modelin başarı oranları ve performansı sunulmuştur."
        },
        {
            "filename": "CountOfAttackTypeAndPercentage.jpg",
            "description": "İkinci veri setinde saldırı türlerinin sayısal dağılımı ve yüzdesel oranları gösterilmiştir."
        },
        {
            "filename": "ProtocolDistb.png",
            "description": "İlk veri setindeki sayısal protokol dağılımın yüzde ile gösterilmesi (TCP, UDP)  "
        },
        
    ]
    return render_template("visuals.html", visuals=visuals)



@app.route("/detect", methods=["POST"])
def detect():
    ip = request.json.get("ip")
    interface = "eth0"  # Gerekiyorsa güncelle

    if not ip:
        return jsonify({"error": "Lütfen geçerli bir IP adresi girin."})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as temp:
        pcap_file = temp.name

    try:
        # IP’ye ait trafiği tshark ile 10 saniye boyunca yakala
        tshark_cmd = [
            "tshark", "-i", interface,
            "-a", "duration:10",
            "-f", f"host {ip}",
            "-w", pcap_file
        ]
        subprocess.run(tshark_cmd, timeout=15)

        if not os.path.exists(pcap_file):
            return jsonify({
                "attack_type": None,
                "error": "Tshark dosyayı oluşturamadı."
            })

        # pcap dosyasından frame length ve ip proto bilgisini çek
        summary_cmd = [
            "tshark", "-r", pcap_file, "-T", "fields",
            "-e", "frame.len", "-e", "ip.proto", "-E", "separator=,"
        ]
        output = subprocess.check_output(summary_cmd).decode("utf-8")

        packets = []
        for line in output.strip().split("\n"):
            try:
                length, proto = line.strip().split(",")
                if length and proto:
                    packets.append({
                        "length": int(length),
                        "protocol_code": int(proto)
                    })
            except ValueError:
                continue

        if not packets:
            return jsonify({
                "attack_type": None,
                "error": "Yeterli trafik verisi yakalanamadı."
            })

        df = pd.DataFrame(packets)

        # Model ile tahmin
        predictions = model.predict(df[["length", "protocol_code"]])
        prediction_series = pd.Series(predictions)

        # En sık görülen saldırı tipini al
        most_common_attack = prediction_series.mode()[0]
        prediction_counts = prediction_series.value_counts(normalize=True) * 100
        confidence = round(prediction_counts[most_common_attack], 2)

        return jsonify({
            "attack_type": most_common_attack,
            "confidence_percent": confidence,
            "total_packets_analyzed": len(predictions)
        })

    except subprocess.TimeoutExpired:
        return jsonify({
            "attack_type": None,
            "error": "Tshark işlem süresi doldu."
        })
    except Exception as e:
        return jsonify({
            "attack_type": None,
            "error": f"Hata oluştu: {str(e)}"
        })
    finally:
        if os.path.exists(pcap_file):
            os.remove(pcap_file)

if __name__ == "__main__":
    app.run(debug=True)
