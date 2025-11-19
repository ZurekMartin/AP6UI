import axelrod as axl
import matplotlib.pyplot as plt

class MyStrategy(axl.Player):
 """
 Strategie s adaptivním prahem pro defekci.
 Začíná spoluprací a následně sleduje chování soupeře.
 Práh pro defekci se dynamicky upravuje na základě nedávného chování soupeře:
 - Pokud soupeř často zrazuje (>30%) v posledních 20 tazích, práh se snižuje
 - Pokud soupeř zřídka zrazuje (<10%) v posledních 20 tazích, práh se zvyšuje
 - Defekce nastane, když celkový poměr zrad soupeře překročí aktuální práh
 
 Names:
 - MyStrategy: Martin Zurek
 """
 name = "MyStrategy"
 classifier = {
     "memory_depth": float("inf"),
     "stochastic": False,
     "long_run_time": False,
     "inspects_source": False,
     "manipulates_source": False,
     "manipulates_state": False
 }
 threshold = 0.20  # Počáteční práh pro defekci
 
 def strategy(self, opponent):
     """
     Začíná spoluprací. Po 20 tazích upravuje práh defekce na základě
     nedávného chování soupeře. Nakonec porovnává celkový poměr zrad 
     soupeře s aktuálním prahem a rozhoduje, zda spolupracovat nebo zradit.
     """
     if not opponent.history:
         return axl.Action.C  # První tah je vždy spolupráce
     
     if len(opponent.history) > 20:
         # Analyzuje posledních 20 tahů soupeře
         recent = opponent.history[-20:]
         recent_defect_ratio = recent.count(axl.Action.D) / len(recent)
         
         # Upravuje práh na základě nedávného chování
         if recent_defect_ratio > 0.3:
             # Snižuje práh, pokud soupeř často zrazuje
             MyStrategy.threshold = max(0.05, MyStrategy.threshold - 0.01)
         elif recent_defect_ratio < 0.1:
             # Zvyšuje práh, pokud soupeř zřídka zrazuje
             MyStrategy.threshold = min(0.25, MyStrategy.threshold + 0.01)
             
     # Rozhoduje, zda spolupracovat nebo zradit na základě celkové historie
     defect_ratio = opponent.defections / len(opponent.history)
     if defect_ratio > MyStrategy.threshold:
         return axl.Action.D  # Zrazuje nad prahem
     return axl.Action.C  # Jinak spolupracuje

players = [
    axl.TitForTat(),
    axl.Cooperator(),
    axl.Defector(),
    axl.CyclerDDC(),
    axl.CyclerCCD(),
    axl.Random(),
    axl.Forgiver(),
    axl.TwoTitsForTat(),
    axl.Grudger(),
    MyStrategy()
]

tournament = axl.Tournament(players, turns=200, repetitions=1, seed=21)
results = tournament.play()

print("\nVýsledné pořadí strategií:")
for name in results.ranked_names:
    if name == "MyStrategy":
        print(f"{results.ranked_names.index(name)+1}. {name} <--- Má strategie")
    else:
        print(f"{results.ranked_names.index(name)+1}. {name}")

summary = results.summarise()

print(f"\n{'Pořadí':<10} {'Jméno':<20} {'Průměrně let volnosti':<25} {'Míra spolupráce'}")
print("="*73)

for player in summary:
    median_score = round(player.Median_score, 4)
    cooperation_rating = round(player.Cooperation_rating, 4)
    print(f"{player.Rank:<10} {player.Name:<20} {median_score:<25} {cooperation_rating}")

plot = axl.Plot(results)

fig, axs = plt.subplots(1, 2, figsize=(14, 10))

ax1 = axs[0]
bp = plot.boxplot(ax=ax1)
ax1.set_title("Boxplot výsledků strategií (skóre = průměr let volnosti)")
ax1.set_xlabel("Strategie")
ax1.set_ylabel("Let volnosti (průměr)")

ax2 = axs[1]
wp = plot.winplot(ax=ax2)
ax2.set_title("Distribuce vítězství pro jednotlivé strategie")
ax2.set_xlabel("Strategie")
ax2.set_ylabel("Počet vítězství")

plt.tight_layout()
plt.show()
