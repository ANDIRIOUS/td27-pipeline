# Posesión cuadro a cuadro: evidencia + Viterbi de tres estados

## Reproducir

Necesitas el paquete de datos en `data/` (nunca se commitea). Un solo script, `solution/run.sh`,
corre todo el pipeline: genera la entrega, la valida, imprime su hash, la puntúa en la ventana
pública y corre el diagnóstico sin etiquetas. Se puede correr de dos formas.

**Con Docker Compose (recomendado).** Solo necesitas Docker; `compose.yaml` define dos
contenedores:

| servicio | para qué | comando |
|---|---|---|
| `pipeline` | Generar la entrega. Es el entorno mínimo y fijado del que depende el hash. | `docker compose run --rm pipeline` |
| `dev` | Explorar, ver el video y hacer figuras: Jupyter con OpenCV, matplotlib y kloppy. | `docker compose up dev` |

- `pipeline` escribe `evaluation/results/Tiki-Data.csv` con la v7, la versión congelada. `TEAM=otro`
  cambia el nombre del archivo y `VERSION=v5` regenera una versión anterior (igual con v1, v2 y v3).
- `dev` imprime en el log un link `http://127.0.0.1:8888/lab?token=…`: ábrelo en el navegador.
  Jupyter solo escucha en tu máquina, no en la red. Para correr un script suelto:
  `docker compose run --rm dev python <script>.py`.
- Si tus datos no están en `./data`, apunta a ellos con `TIKI_DATA=/ruta/a/data`.
- La entrega nunca depende del contenedor `dev`. Sus dependencias están en `requirements-dev.txt`
  y se pueden agregar sin cambiar el hash.

**Con Docker, sin Compose:**

```bash
docker build -f solution/Dockerfile -t mtah-v1:1 .
docker run --rm -v "$PWD":/r -e TEAM=<team_name> mtah-v1:1
```

**Sin Docker** (requiere Python 3.10 o más reciente):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r solution/requirements.txt
TEAM=<team_name> sh solution/run.sh
```

La entrega queda en `evaluation/results/<team_name>.csv` (gitignored). La primera corrida parsea el
ATD y lo cachea en `data/cache/` (también gitignored). Por defecto sale la v7, la versión congelada;
`VERSION=v5` regenera una versión anterior (igual con v1, v2 y v3).

| versión | SHA-256 esperado |
|---|---|
| **v7** (por defecto, congelada: v5 sin evidencia de balón faltante) | `aff8290262161f5fb4bb96897e320eacebab03e0ec43420173af2052ef235ab9` |
| v5 | `b7a2f894600e512aef3bdf108e58fab76ca49a17922f37b575a3d4b0fe4d4389` |
| v3 | `059cdc2d144391ec95f43c3420af3d325f4fc0c452f6b2a20750b65cad2afa60` |
| v2 | `0677dda7f0527b05cbbd2d6afbb96df85a3714018ce14bb4ffe32910d8c997c5` |
| v1 | `463dcdf8badc4f6babbcd11825264ac861caa38bcb2df5af40176a6b99473418` |

Los hashes de v1 y v2 salen iguales en Docker arm64 y en Docker x86_64 (`--platform linux/amd64`),
siempre desde un caché vacío; lo mismo se verificó para la v7. La v1 también se verificó sin Docker, en macOS con Python 3.10. Sube el
archivo exacto que generes y compara su hash con este antes de hacerlo.

| archivo | qué hace |
|---|---|
| `atd.py` | lee el ATD crudo a numpy (verificado: el jugador más cercano coincide 100% con `baseline_raw`) |
| `v1.py` | evidencia por cuadro + Viterbi; también define los bloques que reutiliza la v2 |
| `v2.py` | v1 + clasificador colectivo + reinicio anclado |
| `diagnose.py` | métricas sin etiquetas sobre el partido completo, para comparar versiones |
| `v3.py` | v2 con el peso de las fases elegido por la prueba de huecos (A1) |
| `v5.py` | v3 con los saques iniciales anclados por el juez de reinicios |
| `muertos.py` | barrido de la evidencia de balón faltante; su `run(missing)` es la v5 con otra constante |
| `v7.py` | v5 sin la evidencia de balón faltante (versión congelada) |
| `cambios.py` | estudio del costo de cambio de equipo: recuperaciones tragadas contra cambios inventados |
| `reinicios.py` | juez de reinicios: el golpe del balón en saques de centro, córners y saques de meta |
| `huecos.py` | prueba de huecos simulados: adivinar el equipo sin balón, sin depender de las fases |
| `fronteras.py` | estudio de análisis (no forma parte de la entrega): auditoría de los cortes entre equipos |
| `arrastre.py` | estudio de análisis (no forma parte de la entrega): arrastre del balón suelto, la v6 descartada |
| `run.sh` | pipeline completo; es el comando por defecto del contenedor |
| `Dockerfile`, `requirements.txt` | entorno fijado: Python 3.10.20, numpy 2.2.6, pandas 2.3.3 |

## Versión congelada: v7

El equipo congeló la v7 el 12 nov 2026, con huella `aff82902…235ab9` (arm64 y amd64). El detalle
está en [v7: la v5 sin la evidencia de balón faltante](#v7-la-v5-sin-la-evidencia-de-balón-faltante).

- **Por qué la v7 y no la v5.** La regla fijada antes de correr en `muertos.py` eligió 0. En la
  prueba de huecos, el equipo correcto sube de 62.0% a 65.4% y el juego vivo marcado como muerto
  baja de 16.9% a 11.8%. Las guardas de balón muerto seguro no se mueven.
- **La señal en contra.** Su `D` queda en 39.8%, apenas debajo del rango de 40% a 50% que la
  investigación de José Roberto propone como control de cordura. Es una alarma, no un criterio de
  elección.
- **No entra el arrastre del balón suelto (v6, estudio en `arrastre.py`).** Su referencia usa la misma idea que el
  arreglo, y el Smart Tagging dice lo contrario.
- **No entra el balón del broadcast (`broadcast_sync.py`).** Solo rellena el 8.7% de los cuadros sin balón (5.2%
  en huecos de más de 10 s), y donde ambos ven el balón, el 11.4% difiere en más de 5 m.

## La idea (v1)

El baseline acierta bien el equipo con balón vivo, pero solo marca `D` cuando el tracker pierde el
balón, y el balón se pierde también en juego vivo (huecos con mediana de 10 s). Como cada `D` mal
puesto parte una posesión en dos, y spellF1 cuenta posesiones, el problema central es el balón
muerto.

**1. Evidencia por cuadro**, en nats (log-odds):

- *Equipo:* cuenta solo si el jugador más cercano está a menos de 3 m y el rival más cercano está al
  menos 1 m más lejos. Un balón en el aire o un rebote no aportan nada, y en ese caso se mantiene al
  último poseedor, como pide la definición de posesión.
- *Muerto por la dinámica colectiva:* la mediana de velocidad de los jugadores, suavizada a 2 s.
  Cuando cae por debajo de ~1.3 m/s, los jugadores están caminando. En el partido, el 25% de los
  cuadros calificados tiene la mediana por debajo de 1 m/s, y ahí el balón solo se ve el 22% del
  tiempo. Esta señal funciona aunque falte el balón.
- *Muerto por geometría:* el balón se ve fuera del campo (más de ~0.5 m tras la línea) o salió y no
  se ha vuelto a ver rodando dentro. También cuentan el balón quieto (colocado para un reinicio),
  como indicio de muerto, y el balón golpeado a más de 8 m/s dentro del campo, como indicio de vivo.
- *Smart Tagging:* en cada `SET PIECES`, la ventana [start−4 s, start+3 s] cuenta como muerto y
  [start+4 s, start+7 s] como vivo (el reinicio cae ~3 s después del `start`; lo comprobamos en
  ambos saques iniciales). Las fases con equipo aportan un empujón leve (0.1 nats por cuadro) hacia
  el equipo que ataca. En las fases `DEFENDING…`, el `Team` es quien defiende.

**2. Viterbi con costos de cambio asimétricos.** Cambiar de equipo cuesta 20 nats, o sea ~0.8 s de
control claro: un toque o un desvío no bastan. Pasar de un equipo a `D` también cuesta 20 nats, lo
que exige varios segundos de señal muerta. Salir de `D` cuesta 2, porque todo reinicio tiene un
poseedor claro. Viterbi encuentra la secuencia óptima de toda la mitad, y en la práctica funciona
como una histéresis sin tener que fijar ventanas a mano.

**3. Detalles que mueven el puntaje:**

- Entre el inicio del segundo tiempo calificado (110808) y el saque (~113742) va `D` explícito. Si
  pusiéramos `X`, el scorer rellenaría hacia atrás con el equipo que saca y la primera posesión
  empezaría dos minutos antes.
- Ningún cuadro calificado lleva `X`. Para spellF1, `X` equivale a repetir la etiqueta anterior, y
  escribir esa misma etiqueta da el mismo spellF1 sin regalar macroF1, que es el desempate.

## Lo que agrega la v2

1. **Clasificador colectivo para los cuadros sin control claro.** Es una regresión logística ridge
   entrenada por Newton-Raphson: determinista y solo con numpy. Usa 18 features de la forma de los
   bloques, sin el balón, suavizadas a 2 s:
   - por equipo: centroide, amplitud en x y en y, velocidad hacia el arco rival, velocidad media y
     jugadores más retrasado y más adelantado;
   - además, los contrastes entre los dos bloques.

   Se entrena con **pseudo-etiquetas**, los ~40 mil cuadros con control claro y jugadores en
   movimiento, y se aplica solo donde no hay control claro, que son justo los cuadros con los que
   no se entrenó.
   - En validación cruzada por bloques de 5 min acierta el 77% de las pseudo-etiquetas (log-loss
     0.51 contra 0.67 de la base).
   - Entra con un peso de 1/25 por cuadro: con features suavizadas a 2 s hay ~1 observación
     independiente por segundo.
   - El log-odds se recorta a ±2, porque en la validación cruzada las probabilidades extremas se
     cumplen ~85%, no 99%.
2. **Reinicio anclado.** El "vivo" después de un `SET PIECES` empieza en el primer movimiento real
   dentro de los 20 s siguientes: el balón rueda dentro del campo o los jugadores pasan de caminar a
   correr. Ya no se usa un desfase fijo.
3. **El "pendiente" caduca** también cuando los jugadores pasan de caminar a correr.

## Resultados

Partido completo, sin etiquetas (`diagnose.py`). "Sin balón" es el acuerdo con las fases del Smart
Tagging solo en cuadros vivos sin balón visible, que es justo donde la v1 tenía que adivinar:

| | posesiones | mediana | % D | acuerdo fases | sin balón | reinicios bien |
|---|---|---|---|---|---|---|
| smooth50 | 594 | 4.6 s | 32.3% | 64.3% | 48.1% | 55.9% |
| v1 | 223 | 10.5 s | 42.6% | 84.7%* | 93.7%* | 69.5% |
| **v2** | 231 | 10.1 s | 41.6% | 85.1%* | 94.5%* | **74.6%** |
| *ablaciones sin fases ST (comparación honesta):* | | | | | | |
| v1 | 248 | 9.5 s | 44.4% | 69.5% | 47.7% | 69.5% |
| v1 + colectivo | 252 | 9.3 s | 44.2% | 70.8% | 52.4% | 69.5% |
| v1 + anclado | 253 | 9.3 s | 44.1% | 69.6% | 47.7% | 76.3% |
| v2 completa | 258 | 9.0 s | 43.9% | 71.1% | **54.9%** | 74.6% |

\* Circular: estas versiones usan las fases como evidencia.

Hallazgos que salen de esta tabla:

- **El clasificador colectivo es ganancia limpia.** Sube el acuerdo sin balón ~6 puntos y casi no
  cambia el número de posesiones.
- **El reinicio anclado tiene un costo.** Ubica mejor los reinicios (+5 puntos), pero en el partido
  completo las posesiones de menos de 5 s entre dos `D` pasan de 16 a 21. Algunas pueden ser
  falsos arranques: los jugadores trotan antes de que se reanude.
- **Las fases del Smart Tagging también son ruidosas.** Coinciden con el control claro solo en el
  72% de los cuadros, y cubren el 68% de los cuadros calificados. Con un peso de 0.1 por cuadro
  dominan los huecos sin balón; el clasificador colectivo, con ≤0.04, casi no se nota ahí.

Ventana pública de 5 min y 17 posesiones (`run_evaluation.py --bootstrap 1000`). Sirve solo para
verificar, no para elegir:

| | spellF1 | IC 5–95% | macroF1 | recall D |
|---|---|---|---|---|
| smooth50 | 0.3256 | 0.21–0.45 | 0.6207 | 0.21 |
| v1 | 0.3625 | 0.19–0.54 | 0.8146 | 0.74 |
| v2 | 0.3515 | 0.18–0.52 | 0.8003 | 0.67 |

v1 y v2 difieren en el equivalente a una posesión, muy dentro del ruido. En la ventana se ve el
mecanismo del costo del anclaje: una posesión de 3 s dentro de un balón muerto. La prueba real es
la calificación del partido completo, donde el piso de smooth50 es 0.2433.

## Prueba de huecos simulados (`huecos.py`)

**Por qué hacía falta.** Para comparar versiones sin etiquetas, `diagnose.py` mide, entre otras
cosas, el acuerdo con las fases del Smart Tagging. Pero las versiones usan esas mismas fases como
evidencia: es circular. Si a una versión le bajamos el peso de las fases, esa métrica baja por
construcción, aunque la versión sea mejor. Para decidir cuánto deben pesar las fases (A1) y para
elegir la versión final (A3) hacía falta una referencia que no dependiera de ellas.

**La idea.** El problema difícil son los huecos del tracker en juego vivo: el balón desaparece unos
segundos y el modelo tiene que adivinar quién lo tiene. Esta prueba fabrica huecos donde sí sabemos
la respuesta. Se esconde el balón en tramos donde había control claro, se corre el modelo completo
como si el tracker lo hubiera perdido ahí y se compara lo que dice con el control claro escondido.
Nadie etiqueta nada: la referencia es la misma regla de 3 m y 1 m que usa el modelo.

**Cómo está armada**, con cada decisión fijada antes de correrla:

- **Largos realistas.** Se toman de los 49 huecos reales en juego vivo de al menos 5 s (mediana
  8.9 s, p90 22.7 s). Los huecos más cortos casi no importan, porque el Viterbi mantiene al último
  equipo, y además no dejarían nada que medir tras quitar los bordes.
- **Hueco completo, no solo "sin equipo".** El balón se pone como faltante, así que todo lo que
  depende de él reacciona como en un hueco real: control, balón fuera, quieto o golpeado. El
  clasificador colectivo tampoco puede entrenar con esos cuadros, porque ya no tienen control claro.
- **Solo juego vivo**: tramos con los jugadores corriendo en promedio y al menos 1 s de control
  claro.
- **Se esconde el 12% de lo calificado en cada réplica.** Así el clasificador conserva casi todos
  sus ejemplos de entrenamiento.
- **Bordes fuera.** No se miden los 2 s de cada borde, porque las features se suavizan a 2 s y ahí
  todavía ven datos sin hueco.
- **Monte Carlo.** Son 20 réplicas con semilla fija (2027), así que la salida es idéntica en cada
  corrida. Todas las versiones ven los mismos huecos en cada réplica, y por eso la diferencia contra
  la v2 se mide réplica por réplica.

**Resultados**, en cuadros con control claro escondido (media ± desviación entre réplicas):

| versión | equipo correcto | muerto falso | equipo vs v2, pareado |
|---|---|---|---|
| v1 sin fases | 53.7% ± 3.7 | 20.4% ± 3.8 | −8.6 ± 4.6 |
| v1 | 60.5% ± 3.9 | 15.7% ± 4.0 | −1.8 ± 2.1 |
| v2 sin fases | 59.4% ± 3.6 | 19.4% ± 3.6 | −2.9 ± 2.7 |
| **v2** | **62.3% ± 3.7** | **15.1% ± 4.1** | 0 |

Con 20 réplicas, el error estándar de cada diferencia pareada es su desviación entre √20 (≈ 0.5–1
punto). Las cuatro versiones se separan con claridad:

- El clasificador colectivo aporta ~6 puntos sin fases y ~2 con fases.
- Las fases aportan ~7 puntos en la v1 y ~3 en la v2: el clasificador ya cubre parte de lo que
  ellas daban.
- En un hueco largo, todas las versiones declaran muerto un 15–20% de juego vivo. Es el siguiente
  error grande a atacar.

**¿Es justa con las fases?** Metrica infiere el Smart Tagging del mismo video y tracking que
nosotros. En los huecos simulados el tracker sí vio el balón, así que las fases tuvieron esa
información; en un hueco real quizá no.

Para medirlo, se cuenta cuántas veces cambian de equipo las fases por minuto:

- dentro de huecos reales: 0.62 cambios por minuto, en 10 min (~6 cambios);
- en juego con balón visible: 0.93 cambios por minuto, en 47 min.

Van en la dirección de fases algo "a ciegas" en los huecos reales, pero con 6 cambios la diferencia
cabe en el ruido (±0.25 por minuto). Conclusión: la prueba ordena versiones, pero puede darle a las
fases algo más de crédito del que merecen. Por eso, al elegir su peso con esta prueba se toma el más
bajo que quede dentro de un error estándar del mejor, no el máximo.

**Qué no es.** No es la métrica oficial ni sustituye a spellF1: mide solo la parte de adivinar el
equipo en huecos, no los cortes por balón muerto ni los reinicios. Por eso se usa junto con
`diagnose.py`, no en su lugar. Corre en ~1.5 min:
`docker run --rm -v "$PWD":/r mtah-v1:1 python solution/huecos.py`, o con
`docker compose run --rm pipeline python solution/huecos.py`.

### Peso de las fases (A1): `huecos.py --barrido`

**La hipótesis.** En v1 y v2, cada cuadro dentro de una fase suma 0.1 nats, como si cada cuadro
fuera una observación independiente. Una fase mediana dura ~22 s, así que una sola fase sumaba ~54
nats, más que dos cambios de posesión. La sospecha era que las fases pesaban de más.

**Cómo se decidió.** Antes de correr nada se fijaron, en un commit aparte, una rejilla de ocho
pesos de 0 a 0.1 y la regla de elección: el peso más bajo cuya diferencia pareada contra el mejor
no sea peor que un error estándar. La rejilla se ancla en tres maneras de contar observaciones
independientes:

- una por fase: 0.002;
- una cada 3.7 s, el tramo medio de acuerdo de las fases con el control claro: 0.008;
- una por segundo: 0.03.

| peso | 0 | 0.002 | 0.004 | 0.008 | 0.015 | 0.03 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|---|---|
| equipo correcto en huecos | 59.4% | 59.7% | 60.3% | 60.5% | 61.1% | 61.7% | 62.0% | 62.3% |
| vs 0.1, pareado | −2.9 | −2.5 | −1.9 | −1.8 | −1.1 | −0.6 | −0.2 | 0 |
| posesiones (partido completo) | 258 | 258 | 257 | 252 | 247 | 242 | 239 | 231 |

**Qué salió.** La hipótesis no se sostuvo: en los huecos, más peso siempre ayuda un poco, hasta el
máximo de la rejilla. La regla elige **0.05**, que queda a 0.2 puntos de 0.1, dentro del ruido. Se
prefiere la mitad del peso porque la prueba puede darle a las fases más crédito del que merecen.

**Se revisó también el lado del balón muerto.** En los cuadros donde es seguro, el peso casi no
cambia nada:

| peso | balón fuera | jugadores a menos de 1 m/s | 4 s antes de un reinicio |
|---|---|---|---|
| todos (0 a 0.1) | 96–98% en `D` | 97–98% en `D` | 83–84% en `D` |

Así que subir el peso no tapa muertos reales; lo que hace es unir posesiones (de 258 a 231). Sin
etiquetas no se puede saber si esas uniones son correctas, y es otra razón para no irse al máximo.

**v3 contra v2** (`diagnose.py`, partido completo):

| | posesiones | mediana | % D | acuerdo fases | sin balón | reinicios bien |
|---|---|---|---|---|---|---|
| v2 | 231 | 10.1 s | 41.6% | 85.1%* | 94.5%* | 74.6% |
| v3 | 239 | 10.4 s | 42.9% | 81.3%* | 88.1%* | 74.6% |

\* Circular. Bajan porque las fases pesan la mitad, no porque la v3 sea peor: justo por eso existe
la prueba de huecos.

El hash de la v3 sale igual en Docker arm64 y amd64. No se puntuó en la ventana pública.

## Juez de reinicios (`reinicios.py`)

**Por qué hacía falta.** `diagnose.py` cuenta un reinicio como "bien" si hay `D` cerca del inicio
del tag y juego vivo entre 5 y 8 s después: supone de entrada que el juego se reanuda ~3 s después
del tag. Con esa métrica no se puede juzgar una regla que mueve el reinicio. El primer intento de A2
(la v4) bajó de 74.6% a 57.6% en ella, y no había forma de saber si la v4 era peor o si la
métrica la castigaba por construcción.

**La idea.** En tres tipos de balón parado, el reglamento fija dónde está el balón: el saque de
centro en el punto central, el córner en una esquina y el saque de meta en el área chica. El juez
busca el primer arranque del balón (pasa de 5 m/s) que ocurre en ese lugar, desde 5 s antes del tag
hasta 20 s después. Ese es el golpe, y sirve de referencia sin suponer ningún desfase. Quedan fuera
los saques de banda (el balón va en las manos) y los tiros libres (pueden ser en cualquier lugar,
así que el lugar no confirma nada).

**Una primera versión falló, y se cuenta.** Buscaba el balón quieto y luego pateado, y no detectó
ninguno de los 33 balones parados: el ATD casi nunca ve el balón colocado. El tracker lo pierde
mientras está quieto, o lo ve moverse mientras lo acomodan. Se cambió la detección antes de medir
cualquier regla contra el juez (commits `a4aaac7` y `f5cc486`).

**Qué detecta.** 17 golpes de 28:

| tipo | detectados |
|---|---|
| córners | 5 de 12 |
| saques de meta | 8 de 10 |
| saques de centro | 4 de 6 |

- 12 de los 17 son "fuertes": el balón pasa de 10 m/s en medio segundo. Un arranque que apenas cruza
  5 m/s puede ser el portero acomodando el balón.
- El saque del 2º tiempo no se detecta, y es lo correcto: su "balón" es el artefacto que cruza 40 m
  a velocidad constante, y nunca arranca desde el centro.
- El golpe cae a una mediana de **+4.4 s** del inicio del tag. En los saques de centro, a +3.1 a
  +3.7 s.

**Error de cada regla contra el golpe**, en segundos (positivo = tarde):

| regla | todos (17): mediana | dentro de 1 s | fuertes (12): mediana | dentro de 1 s |
|---|---|---|---|---|
| tag + 3 s fijo | −1.4 | 29% | −2.4 | 33% |
| tag + 5.2 s (mediana de las anclas de v3) | +0.9 | 24% | −0.1 | 33% |
| **ancla de v2 y v3** (cualquier movimiento) | **−0.1** | **82%** | **−0.1** | **83%** |
| ancla de v4 (movimiento sostenido 2 s) | +1.1 | 29% | +0.9 | 42% |
| etiquetas de v3 | −0.1 | 73% | −0.1 | 80% |
| etiquetas de v4 | +1.4 | 27% | +1.0 | 40% |

**Qué se concluye:**

- **La v4 llega tarde.** Ancla ~1 s después del golpe, porque los jugadores tardan en correr. El
  experimento de A2 queda descartado con evidencia, no solo por una métrica sesgada.
- **El "~3 s después del tag" solo vale para los saques de centro.** En córners y saques de meta el
  golpe llega más tarde, así que un desfase fijo no sirve.
- **El ancla de v2 y v3 cae sobre el golpe** en 4 de cada 5 casos.

**Cuidado con lo que no mide.** El ancla de v3 y el juez usan la misma señal: el balón rodando. Que
coincidan cuando el balón se ve es casi esperable. El juez confirma que el ancla encuentra el golpe
cuando el balón es visible, pero no puede juzgar los reinicios sin balón visible. Ahí el ancla
depende de que los jugadores arranquen, y ahí siguen los posibles falsos arranques. Esa parte de A2
queda como limitación.

### v5: saques iniciales con el juez

Hasta la v3, el `D` explícito antes de cada saque inicial llegaba al tag + 3 s, un desfase
comprobado a mano mirando esos dos saques. Era la última constante sacada de momentos concretos del
partido. La v5 la cambia por una regla:

1. Si el juez ve el golpe, el `D` llega hasta ahí. En el 1er tiempo lo ve a los +3.7 s.
2. Si no lo ve, se usa la mediana de los saques de centro que sí detectó, calculada en el código. En
   el 2º tiempo no lo ve, por el artefacto, y la mediana da +3.4 s con 4 saques.

El resultado casi no cambia: 13 cuadros distintos de la v3, la primera posesión de cada mitad
empieza 0.1 s y 0.4 s más tarde, y `diagnose.py` da exactamente lo mismo. Lo que cambia es que ya
no queda ninguna constante comprobada a mano.

Como la v5 usa el juez para esos dos saques, el juez ya no sirve para evaluarlos en la v5. Sigue
sirviendo para los córners y los saques de meta, que la v5 no toca. El hash sale igual en Docker
arm64 y amd64.

## Costo de cambio de equipo (`cambios.py`)

**La sospecha.** En la v5 hay 27 tramos de 1 s o más en los que el rival tiene control claro dentro
de una posesión nuestra, y más de la mitad de las posesiones de más de 40 s tienen alguno. Una
recuperación corta del rival obliga al Viterbi a cambiar dos veces (ida y vuelta). Con un costo de
20 nats por cambio, el rival necesita más de ~1.6 s de control claro para que se le reconozca.

**El criterio** sale de la definición del reto: un cambio de posesión exige control, no un toque ni
un desvío. Nuestro indicio de control es un *episodio*: 1 s seguido de control claro. Con él se
cuentan dos errores opuestos:

- **tragados:** episodios del rival dentro de una posesión nuestra;
- **inventados:** posesiones con el balón visible al menos el 80% del tiempo en las que el equipo
  nunca tiene un episodio.

La rejilla de seis costos, el criterio y la regla (menor suma; si hay empate, el costo más alto)
quedaron en un commit aparte antes de correr (`8ca8f70`).

| costo | 20 (v5) | 15 | 12.5 | 10 | 7.5 | 5 |
|---|---|---|---|---|---|---|
| posesiones | 239 | 275 | 311 | 350 | 399 | 502 |
| menos de 2 s | 12 | 22 | 35 | 53 | 86 | 150 |
| tragados | 27 | 16 | 6 | 1 | 0 | 0 |
| inventados | 16 | 27 | 39 | 57 | 95 | 167 |
| **suma** | **43** | **43** | 45 | 58 | 95 | 167 |
| huecos, vs 20 (pareado) | 0 | +0.3 | +0.7 | +0.7 | +1.1 | +1.2 |

**Qué salió.** El costo actual ya está en el punto de equilibrio: la regla elige 20, empatado con 15.

- Bajarlo recupera tragados, pero inventa el mismo número o más de posesiones sin control.
- La prueba de huecos mejora un poco con costos bajos, porque ahí casi no hay control claro que
  inventar; no compensa lo que se pierde con el balón visible.
- No hay v6.
- Los ~27 tragados no se arreglan con este parámetro. Hace falta una pista nueva que distinga una
  recuperación real de un duelo, por ejemplo que después el balón salga del jugador hacia un
  compañero. Queda como siguiente idea.

## v7: la v5 sin la evidencia de balón faltante

**Qué cambia.** Una sola constante: los nats por cuadro sin balón hacia `D` pasan de 0.3 (v1 a v5) a
0. Todo lo demás es la v5. Está en `v7.py`, que reutiliza `muertos.run`.

**Por qué.** No ver el balón no dice nada sobre vivo o muerto: falta en el 41% de lo calificado, en
huecos de mediana 10 s, y no solo cuando el balón está muerto. Con 0.3 por cuadro, un solo hueco
acumulaba ~75 nats hacia `D`, contra 20 que cuesta entrar a `D`. El valor lo eligió la regla fijada
antes de correr en `muertos.py`, no un ajuste a mano. En la prueba de huecos (20 réplicas,
semilla 2027):

| falta | equipo correcto | muerto falso | muerto falso pareado vs 0.3 |
|---|---|---|---|
| 0.3 (v5) | 62.0% | 16.9% | -- |
| 0 (v7) | 65.4% | 11.8% | -5.1 pt (EE 0.34) |

Las guardas de muerto seguro (balón fuera, jugadores caminando, 4 s antes de un `SET PIECES`) y el
detector de reinicios no se mueven.

**`diagnose.py`, partido completo:**

| | posesiones | mediana | % D | acuerdo fases | sin balón | reinicios bien | cortas D-x-D |
|---|---|---|---|---|---|---|---|
| v5 | 239 | 10.4 s | 42.9% | 81.3%* | 88.1%* | 74.6% | 20 |
| v7 | 242 | 10.5 s | 39.8% | 81.8%* | 88.3%* | 74.6% | 17 |

\* Circular, como arriba.

**Por qué v7 y no v6.** La v6 es esta misma versión MÁS el arrastre del balón suelto (estudio descartado, cuyo
código de análisis es `arrastre.py`; no hay `v6.py` en la entrega), que está en disputa. La v7 es la v6 sin arrastre: su CSV es idéntico byte a byte a la variante "sin falta" de
la v6 (`carry_evidence=0.0`, calculada por otro camino). El hash sale igual en Docker
arm64 y amd64, desde caché vacío.

**Dos reservas.**
- La prueba de huecos solo mide un lado del error (el juego vivo declarado muerto). Por el otro lado,
  el `D` baja de 42.9% a 39.8% y aparecen 3 posesiones nuevas en tramos largos sin balón; sin
  etiquetas no se sabe si esas uniones o cortes son correctos.
- No se puntuó en la ventana pública: eso se hace una sola vez, al final.

## Herramientas de desarrollo

El código se escribió con ayuda de Claude Code (Anthropic) como asistente de programación. Ningún
modelo de IA vio el video ni produjo, revisó o corrigió etiquetas: el pipeline nunca abre un video,
solo lee el tracking automático y el Smart Tagging, y cada etiqueta sale de `run.sh`.

## Limitaciones y siguientes pasos

- **Juego vivo marcado como muerto en huecos largos.** La v7 lo bajó de 16.9% a 11.8% en la prueba
  de huecos, pero sigue siendo el error más grande que medimos. El balón del broadcast no alcanza
  para rellenar esos huecos ([Versión congelada: v7](#versión-congelada-v7)).
- **El cambio de equipo cae temprano.** De 144 cambios revisables según nuestra referencia, el
  modelo marca 74, y en esos el corte cae una mediana de 0.88 s antes (`fronteras.py`). El arrastre del
  balón suelto lo corregiría, pero no sabemos dónde corta la respuesta oficial un cambio de equipo,
  y sin etiquetas no se puede resolver.
- **Recuperaciones cortas que no se marcan.** Unas 28 quedan dentro de posesiones nuestras en la v7
  (27 en la v5, la versión que midió `cambios.py`). Bajar el costo de cambio no ayuda; haría falta
  una pista que distinga una recuperación de un duelo.
- **Reinicios sin balón visible.** El juez solo ve los reinicios con balón visible. Donde no se ve,
  el ancla depende de que los jugadores arranquen, y ahí pueden quedar falsos arranques.
- **Portero con el balón en las manos.** Es juego vivo, pero con el balón quieto y todos caminando
  se parece a un muerto. No lo tratamos.
- **Muertos y posesiones muy cortas.** Saques de banda y tiros libres rápidos de 2 a 4 s, y
  posesiones de menos de 1.5 s, casi nunca se detectan, a propósito: cada intento fallido parte una
  posesión real.
- **La prueba de huecos favorece un poco a las fases**, porque en los huecos simulados el Smart
  Tagging sí vio el balón. Por eso pesan 0.05 y no 0.1.
- **Unidades de velocidad.** La columna dice `kph` en el XML, pero sus valores solo son plausibles
  en m/s. Se usa como señal relativa.
