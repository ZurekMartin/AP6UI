# OpenCV Detekce Obličeje

Jednoduchý program v Pythonu využívající OpenCV pro detekci obličeje, očí a úsměvu v obrázcích nebo z webkamery.

## Funkce

- **Detekce obličeje**: Program detekuje obličej pomocí Haar kaskády a zvýrazní jej obdélníkem.
- **Detekce očí**: V rámci detekovaného obličeje nalezne oči a označí je.
- **Detekce úsměvu**: Na spodní polovině obličeje detekuje úsměv a označí jej.
- **Detekce barvy očí**: Analyzuje barvu očí a vypíše přibližný název barvy.
- **Party mód (virtuální brýle)**: Umožňuje v reálném čase přes webkameru přidat na obličej virtuální brýle.

## Režimy

1. **Obrázek**: Uživatel si vybere obrázek ze souboru, který je následně analyzován.
2. **Webkamera**: Detekce se provádí přímo na vstupu z webkamery, včetně možnosti aktivace party módu.

## Ovládání

### Režim Obrázek
- `s` = uložit upravený obrázek
- `q` nebo `ESC` = ukončit program

### Režim Webkamera
- `s` = uložit aktuální snímek
- `g` = zapnout/vypnout party mód (virtuální brýle)
- `q` nebo `ESC` = ukončit program

## Požadavky

- Python 3.x
- OpenCV
- NumPy
