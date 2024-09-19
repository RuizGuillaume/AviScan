import pandas as pd
import random
from datetime import datetime, timedelta


def generate_test_data(num_entries=100):
    data = []
    classes = ["class1", "class2", "class3"]  # Ajoutez ou modifiez selon vos besoins
    start_date = datetime.now() - timedelta(days=30)

    for _ in range(num_entries):
        date = start_date + timedelta(minutes=random.randint(0, 43200))
        predicted_class = random.choice(classes)
        true_class = random.choice(classes)
        confidence = random.uniform(0.5, 1.0)

        data.append(
            {
                "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_class": predicted_class,
                "true_class": true_class,
                "confidence": confidence,
            }
        )

    df = pd.DataFrame(data)
    df.to_csv("logs/performance_logs.csv", index=False)
    print("Données de test générées et sauvegardées dans logs/performance_logs.csv")


if __name__ == "__main__":
    generate_test_data()
