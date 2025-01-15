import matplotlib.pyplot as plt

def plot_pair_density(file_path, parameters):
    """
    Liest die Ergebnisse aus einer Datei und erstellt einen Graphen der paarweisen Dichte.
    
    :param file_path: Pfad zur Datei mit den Ergebnissen.
    :param ringsize: Größe des Rings (Abstandsschritt) in Metern.
    """
    try:
        # Ergebnisse aus der Datei lesen
        with open(parameters, 'r') as file:
            system_size = file.readline().strip()
            N = file.readline().strip()
            steps = file.readline().strip()
            temp = file.readline().strip()
            sigma = float(file.readline().strip())
            ringsize = float(file.readline().strip())
        
        with open(file_path, 'r') as file:
            results = [float(line.strip()) for line in file.readlines()]
        
        # X-Achse (Radien basierend auf der Ringgröße berechnen)
        radii = [ringsize*i/sigma for i in range(len(results))]
        
        # Mittelwert der paarweisen Dichte berechnen
        mean_density = sum(results) / len(results)

        # Graph erstellen
        plt.figure(figsize=(10, 6))
        plt.plot(radii, results, linestyle='-', color='b', label='Paarweise Dichte')
        plt.plot(radii, [mean_density for i in range(len(results))], linestyle='--', color='r', label='Mittelwert')
        plt.title("Radiale Verteilungsfuntion für Systemgröße: " + system_size + " nm, Teilchenanzahl: " + N + "\n" + "Zeitschritte: " + steps + ", Temperatur: " + temp + " K")
        plt.xlabel(r"$r/\sigma$")
        plt.ylabel("g(r)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("pair_density.png")
        # Graph anzeigen
        plt.show()
    
    except FileNotFoundError:
        print(f"Die Datei {file_path} wurde nicht gefunden.")
    except ValueError as e:
        print(f"Fehler beim Lesen der Datei: {e}")

# Beispielaufruf
if __name__ == "__main__":
    filename_1 = "pair_results.txt"
    filename_2 = "parameters.txt"
    plot_pair_density(filename_1, filename_2)
