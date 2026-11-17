# Quellen und Lizenzen der Visuals

Stand: 2026-11-12, ergänzt 2026-11-13 und 2026-11-16 (Neu-Rendering mit korrigierten Namen und Fadenkreuzen, siehe Abschnitt 8).

Die fertigen PNGs, GIF und MP4 liegen nicht in diesem Repository; hier stehen die Skripte, die Koordinaten und diese Quellenliste.
Kein Bild verwendet SomBench-Kacheln (CC BY 4.0, Testset). Kein NASA-Logo.

## Gemeinsame Bildquelle (Mondaufnahmen)

- Produkt: USGS LROC WAC Global Morphology Mosaic, 100 m/Pixel, Juni 2013 (uint8-Darstellung, nicht die rohen I/F-Kanäle).
- Seite: https://astrogeology.usgs.gov/search/map/moon_lro_lroc_wac_global_morphology_mosaic_100m
- Datei: `Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif` (über USGS/PDS, S3).
- Lizenz: gemeinfrei (US-Behörde; USGS-Mosaik). Die zugrunde liegenden LROC-Daten stammen von NASA/GSFC/Arizona State University und sind über das PDS ebenfalls gemeinfrei.
- Pflichtnennung: NASA/GSFC/Arizona State University (LROC WAC), USGS-Mosaik Juni 2013.
- Ausschnitte: jeweils 512 × 512 Pixel des Mosaiks (einfache Zylinderprojektion, Mondradius 1737,4 km, 100 m/Pixel am Äquator). Nord-Süd sind das 51,2 km; Ost-West sind es 51,2 km × cos(Breite), z. B. Clavius rund 27 km, Tycho rund 38 km. Die Ausschnitte sind im Mosaik-Raster gezeigt (nicht entzerrt), so wie das Modell sie gesehen hat. Koordinaten unten je Bild. (Korrigiert 16. Nov. 2026; vorher hieß es pauschal „51,2 km Kantenlänge“.)

## Gemeinsame Hinweis zur Erkennung (für alle Kraterbilder)

- Modell: offenes NASA-IBM Lunar Foundation Model, eigene Nachstellung des Feintunings, Lauf (Startwert) 46, Datei `out1/lunar_real/runs/seed46/ckpt/best.ckpt` (höchste Test-mAP der drei Läufe: 0,2551).
- Schwelle: 0,7 (in den NASA-Beispielen oft 0,5; hier strenger, für ein ruhigeres Bild).
- Das Modell gibt Rechtecke aus. Die goldenen Kreise sind daraus abgeleitet (Mittelpunkt und mittlerer Halbmesser aus Breite und Höhe).
- Eingabe für diese Bilder: das einbandige USGS-Mosaik, standardisiert und auf alle fünf Kanäle übertragen (je Kanal auf dessen Mittelwert und Streuung skaliert), damit es durch die offizielle Normierung läuft. Das ist eine Visualisierung, kein Messlauf und kein Vergleich mit der veröffentlichten mAP. Die mAP 0,2546 kommt aus den Läufen auf den offiziellen fünfkanaligen Testbildern.
- Kreise, die über den Bildrand hinausragen würden, sind auf den veröffentlichten PNGs weggelassen (strenge Randregel: nur Kreise ganz im Ausschnitt), damit nichts abgeschnitten wirkt. Die Rohvorhersagen liegen unter `detections/*_preds.npz`.

## Die einzelnen Dateien

### 1. `infografik-nachgestellt.png` — 2160 × 2700, 4:5

Infografik im Referenzstil (creme, Leiste „KI-EXPERIMENTE“). Blickfang: Ausschnitt Clavius mit Markierungen. Zahlen: eigenes Mittel 0,2546 (drei Läufe 0,2551 · 0,2548 · 0,2539) neben 0,2581 laut NASA-IBM-Modellkarte. Fakten: 0 €, knapp 10 Minuten pro Runde, 30 Runden je Lauf.

- Mondbild: wie oben, Ausschnitt Clavius, Mitte 58,05° S, 13,55° W (Mosaikzeile 44638, Spalte 50217).
- Lizenz Mondbild: gemeinfrei, Nennung NASA/GSFC/ASU, USGS.
- Kreise und Layout: eigenes Werk.

### 2. `highlight-clavius.png` — 2160 × 2700, 4:5

Dieselbe Clavius-Aufnahme, größer, als Einzelbild für eine Karussell-Folie.

- Mitte 58,05° S, 13,55° W. 17 Kreise bei Schwelle 0,7 (2 randüberschreitende Markierungen weggelassen).
- Lizenz: Mondbild gemeinfrei (NASA/GSFC/ASU, USGS); Kreise eigenes Werk.

### 3. `highlight-palus-somni.png` — 2160 × 2700, 4:5

Palus Somni (Ebene zwischen Mare Tranquillitatis und Mare Crisium), Mitte 12,0° N, 44,0° O (Mosaikzeile 23396, Spalte 67668). 51 Kreise bei Schwelle 0,7 (1 randüberschreitende Markierung weggelassen).

- Lizenz: Mondbild gemeinfrei (NASA/GSFC/ASU, USGS); Kreise eigenes Werk.

### 4. `highlight-tycho-nordostteil.png` — 2160 × 2700, 4:5

Nordostteil des Kraters Tycho (Ausschnittmitte rund 29 km nordöstlich der Kratermitte 43,3° S 11,2° W, also innerhalb des Kraters; zu sehen sind Innenwand und Rand). Mitte 42,5° S, 10,5° W (Mosaikzeile 39922, Spalte 51142). 12 Kreise bei Schwelle 0,7 (2 leicht randüberschreitende Markierungen weggelassen).

- Lizenz: Mondbild gemeinfrei (NASA/GSFC/ASU, USGS); Kreise eigenes Werk.

### 5. Quadratische Reinzeichnungen, 2160 × 2160, ohne Rahmen und Titel

Dieselben Aufnahmen, nur Mond und goldene Kreise, ohne Infografik-Rahmen. Nutzbar, wenn ein Bild ohne Text gebraucht wird.

| Datei | Ausschnitt | Mitte | Schwelle | Kreise | Lizenz Mondbild |
|---|---|---|---|---|---|
| `krater-clavius.png` | Clavius | 58,05° S, 13,55° W | 0,7 | 17 | gemeinfrei, NASA/GSFC/ASU, USGS |
| `krater-palus-somni.png` | Palus Somni | 12,0° N, 44,0° O | 0,7 | 51 | gemeinfrei, NASA/GSFC/ASU, USGS |
| `krater-tycho-nordostteil.png` | Tycho, Nordostteil | 42,5° S, 10,5° W | 0,7 | 12 | gemeinfrei, NASA/GSFC/ASU, USGS |
| `krater-copernicus-nordostteil.png` | Copernicus, Nordostteil | 10,2° N, 19,5° W | 0,7 | 9 | gemeinfrei, NASA/GSFC/ASU, USGS |

`krater-copernicus-nordostteil.png` ist ein zusätzlicher, sauberer Ausschnitt (Nordostteil von Copernicus: Boden, Terrassen und Rand; markiert sind kleinere Krater). Nicht als 4:5-Folie gesetzt; die drei stärksten Blickfänge sind Clavius, Palus Somni und Tycho.

## Lizenzgrenzen

- `detections/` enthält keine SomBench-Kacheln.
- LROC Featured Images von lroc.asu.edu sind nicht verwendet (ASU-Copyright).

## 6. Titelbilder (`titel/`)

Stand: 2026-11-13, Fadenkreuze korrigiert 2026-11-16. Skript: `titel/make_titel.py`.

Zwei Varianten (creme, im Referenzstil der anderen Infografiken; dunkel, Mond als Blickfang), jeweils in 4:5 (2160×2700) und als Linkvorschau (1200×627).

| Datei | Format | Variante |
|---|---|---|
| `titel/titel-dunkel-4x5.png` | 2160×2700, 4:5 | dunkel |
| `titel/titel-dunkel-1200x627.png` | 1200×627 | dunkel, Linkvorschau |
| `titel/titel-creme-4x5.png` | 2160×2700, 4:5 | creme, Leiste „KI-EXPERIMENTE“ |
| `titel/titel-creme-1200x627.png` | 1200×627 | creme, Linkvorschau |

### Mondscheibe

- Produkt: NASA Scientific Visualization Studio, „Moon Phase and Libration, 2026“, unbeschriftete Bildfolge (`plain`).
- Seite: https://svs.gsfc.nasa.gov/5587/
- Datei: `moon.7864.tif`, 5760×3240, RGBA mit Alphakanal.
  https://svs.gsfc.nasa.gov/vis/a000000/a005500/a005587/frames/5760x3240_16x9_30p/plain/moon.7864.tif
- Zeitpunkt: 24. November 2026, 15:00 UTC (Bild 7864), Phase 99,81 %, scheinbarer Durchmesser 1986,7″, Entfernung 360.755 km. Sub-Earth-Punkt 6,466° S, 2,461° W, Positionswinkel 347,892°. Bestätigt über die Dial-A-Moon-API: https://svs.gsfc.nasa.gov/api/dialamoon/2026-11-24T15:00
- Datengrundlage der Visualisierung (laut SVS): LROC WAC Color Mosaic, Natural Color Hapke Normalized (NASA/GSFC / Arizona State University), LOLA-Höhenmodell, JPL-DE421.
- Lizenz: gemeinfrei (Werk der US-Behörde; NASA Scientific Visualization Studio).
- Pflichtnennung: NASA’s Scientific Visualization Studio (NASA/GSFC).
- Kein NASA-Logo, keine Beschriftungen aus der „fancy“-Fassung, kein generiertes Mondbild.

### Ausschnitte und Markierungen auf dem Titelbild

- Palus Somni, 12,0° N 44,0° O, 51 Kreise bei Schwelle 0,7: dieselbe Reinzeichnung wie `krater-palus-somni.png` (USGS LROC WAC Global Morphology Mosaic 100 m, Juni 2013, gemeinfrei; Nennung NASA/GSFC/ASU, USGS). Visualisierung, kein Messlauf.
- Tycho, Nordostteil, 42,5° S 10,5° W, 12 Kreise bei Schwelle 0,7: dieselbe Reinzeichnung wie `krater-tycho-nordostteil.png`, gleiche Quelle und Lizenz.
- Die goldenen Fadenkreuze auf der Mondscheibe liegen an den geografisch korrekten Stellen. Rechnung: orthografische Projektion mit dem Sub-Earth-Punkt und dem Positionswinkel des SVS-Bildes 7864. Korrigiert 16. Nov. 2026: Die Fassung vom 13. Nov. drehte mit falschem Vorzeichen des Positionswinkels, die Fadenkreuze lagen dadurch rund ein Viertel des Scheibenradius daneben (Tycho-Fadenkreuz östlich von Tycho). Neu geprüft am Bild selbst: projizierte Mittelpunkte von Copernicus, Kepler, Aristarchus, Tycho, Plato und Proclus liegen auf den Kratern; das Tycho-Fadenkreuz (Ausschnittmitte) liegt im Nordostteil des Kraters.
- Kreise: offenes NASA-IBM-Modell, eigene Nachstellung, Startwert 46, Schwelle 0,7, wie in den übrigen Visuals. Ein weicher Goldschein ist nur eine grafische Verstärkung der vorhandenen Kreise, keine neuen Erkennungen.
- Zahlen auf dem Bild: 0,2546 (Mittel der drei eigenen Läufe) und 0,2581 laut NASA und IBM, dieselben geprüften Werte wie in der README.
- Keine SomBench-Kacheln.

Die Arbeitsdateien unter `titel/src/` (Original-TIFF, ausgeschnittene Scheibe, Zwischengrafiken) und die Vorschaubilder `_preview-*.png` sind nicht zum Posten bestimmt.


## 7. Animiertes GIF `mond.gif` / `mond.mp4` (Skript `animation/mond.py`), Stand 2026-11-16

- Zählregel (gilt für alle Bilder): Score ≥ 0,7 aus den archivierten Vorhersagen `detections/*_preds.npz` (Lauf vom 2026-11-12, Checkpoint Lauf 46); gezeigt und gezählt wird ein Kreis nur, wenn er ganz im 512-Pixel-Ausschnitt liegt. Damit pixelgleich zu `krater-clavius.png` (17), `krater-tycho-nordostteil.png` (12), `krater-palus-somni.png` (51), `krater-copernicus-nordostteil.png` (9).
- Ausschnitte im GIF: Mare Imbrium 9 (von 9), Copernicus Nordostteil 9 (9), Tycho Nordostteil 12 (14), Clavius Kraterboden 17 (19), Südliches Hochland 26° S 27,5° O 105 (114), Petavius Kratermitte 30 (32), Palus Somni 51 (52). Summe 233 gezeigt, 249 Erkennungen ≥ 0,7.
- `final/clean_manifest.json` nannte für Clavius-zentriert, Tycho, Hochland Werte einer früheren, lockereren Randregel (19, 14 …). Am 16. Nov. hat `make_clean.py` alle vier Reinzeichnungen mit der strengen Randregel neu erzeugt (pixelgleich zu den bisherigen Bildern, Zählung 17 / 51 / 12 / 9); die unveröffentlichten Altdateien mit lockerer Regel sind gelöscht. Die Zählung je Ausschnitt (auch für das GIF, Summe 233) steht in `final/clean_manifest.json` unter `mond_animation`.
- Hinweis: Ein erneuter Inferenzlauf mit demselben Checkpoint ist nicht bitgleich; Erkennungen knapp an der Schwelle (Score 0,70–0,71) kippen dabei um ±1 pro Ausschnitt. Maßgeblich sind die archivierten `_preds.npz`.
- Mondkugel: NASA SVS CGI Moon Kit, `lroc_color_poles_4k.tif` (https://svs.gsfc.nasa.gov/4720/), gemeinfrei, LROC WAC (NASA/GSFC/ASU). Lage geprüft: Ausschnitte byte-gleich aus dem USGS-Mosaik neu geladen, Georeferenz aus dem GeoTIFF-Header; Textur gegen Mosaik korreliert, Restabweichung 0–5 km (unter 1 Pixel auf der Kugel).

## 8. Neu-Rendering 2026-11-16

- Namen korrigiert: „Hochland“ → Palus Somni (12° N 44° O, rund 70 km von der Mitte von Palus Somni, 14,1° N 45,0° O); „Umgebung Tycho“ → Tycho, Nordostteil; „Rand von Copernicus“ → Copernicus, Nordostteil.
- Kein „eigenes Modell“ mehr auf den Bildern; Nennung jetzt „offenes NASA-IBM-Modell, eigene Nachstellung, Schwelle 0,7“.
- Folien zeigen die Zahl der Kreise in der Unterzeile (Clavius 17, Palus Somni 51, Tycho 12).
- Titelbilder: Fadenkreuze mit korrigiertem Positionswinkel (siehe Abschnitt 6); Linkvorschau-Fußzeile ohne Uhrzeit, damit sie nicht an den Rand stößt.
- Dateinamen neu: `highlight-palus-somni.png`, `highlight-tycho-nordostteil.png`, `krater-palus-somni.png`, `krater-tycho-nordostteil.png`, `krater-copernicus-nordostteil.png`. Alte Fassungen gelöscht.
- Die Ausschnitt-Kennungen (Dateinamen unter `crops/` und `detections/`) bleiben unverändert: `hochland-ost` = Palus Somni, `tycho-umfeld` = Tycho, Nordostteil, `copernicus-rand` = Copernicus, Nordostteil, `clavius-zentriert` = Clavius, Kraterboden, `theophilus-umfeld` = Südliches Hochland (26° S 27,5° O), `petavius-umfeld` = Petavius, Kratermitte, `mare-scherben` = Mare Imbrium. `clavius-umfeld` und `hochland-15w` werden in keinem veröffentlichten Bild verwendet.
