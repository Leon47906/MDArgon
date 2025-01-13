import matplotlib.pyplot as plt

def plot_pair_density(file_path, ringsize):
    """
    Liest die Ergebnisse aus einer Datei und erstellt einen Graphen der paarweisen Dichte.
    
    :param file_path: Pfad zur Datei mit den Ergebnissen.
    :param ringsize: Größe des Rings (Abstandsschritt) in Metern.
    """
    try:
        # Ergebnisse aus der Datei lesen
        with open(file_path, 'r') as file:
            results = [float(line.strip()) for line in file.readlines()]
        
        # X-Achse (Radien basierend auf der Ringgröße berechnen)
        radii = [i * ringsize for i in range(len(results))]
        
        # Mittelwert der paarweisen Dichte berechnen
        mean_density = sum(results) / len(results)

        # Graph erstellen
        plt.figure(figsize=(10, 6))
        plt.plot(radii, results, linestyle='-', color='b', label='Paarweise Dichte')
        plt.plot(radii, [mean_density for i in range(len(results))], linestyle='--', color='r', label='Mittelwert')
        plt.title("Paarweise Dichte gegen Radius")
        plt.xlabel("Radius (m)")
        plt.ylabel("Normierte Paar-Dichte")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        # Graph anzeigen
        plt.show()
    
    except FileNotFoundError:
        print(f"Die Datei {file_path} wurde nicht gefunden.")
    except ValueError as e:
        print(f"Fehler beim Lesen der Datei: {e}")

# Beispielaufruf
if __name__ == "__main__":
    file_path = "pair_results.txt"
    ringsize = 1e-10  # Muss mit dem Ringgrößenparameter im C++-Code übereinstimmen
    plot_pair_density(file_path, ringsize)
