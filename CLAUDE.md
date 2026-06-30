# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Что это

**QuickLabel** — локальное web-приложение для быстрой разметки датасетов с ускорением через **SAM 2** (интерактивно) и **SAM 3** (авторазметка по текстовому запросу), плюс страница локального **обучения** (RF-DETR / YOLO11) с живым дашбордом метрик. Изначально создано на основе VisoLabel, но теперь полностью автономно: содержит свою копию SAM-бэкенда (`QuickLabel/ml_backend`) и веса моделей (`QuickLabel/models/*.pt`). Папки `VisoLabel/`, `analyz/`, `wheels/`, `models/` в gitignore — весь исходный код лежит в `QuickLabel/`.

UI — это обычные HTML/CSS/JS в `QuickLabel/web/`, **без шага сборки**. Backend — FastAPI (`python -m backend.server`). Бо́льшая часть пользовательских строк и README на русском языке.

## Команды

Все команды запускаются из папки `QuickLabel/`. Требуется **Python 3.13**.

```powershell
.\setup.ps1                  # один раз: создать .venv, поставить зависимости + встроенные wheel'ы SAM (CUDA 12.4)
.\setup.ps1 -CpuOnly         # torch только под CPU
.\setup.ps1 -WithTraining    # дополнительно поставить rfdetr==1.5.2 + ultralytics для страницы обучения
.\run.ps1                    # запуск сервера (автопоиск .venv, открывает браузер на http://127.0.0.1:8765)
.\build_exe.ps1              # пересборка лёгкого лаунчера QuickLabel.exe (нужен pyinstaller)
```

В репозитории **нет тестов, линтера и CI**. Запустить сервер напрямую: `& .venv\Scripts\python.exe -m backend.server`.

venv большой (~5 ГБ с torch) и **непереносимый** — пересоздавайте его через `setup.ps1` на каждой машине. `rfdetr` закреплён на `1.5.2`, потому что в 1.7+/1.8 изменился train-API, который использует тренер.

## Архитектура

### Модель процессов — ключевое архитектурное решение

Web-сервер **никогда не запускает torch/CUDA в своём процессе**. Тяжёлая ML-работа идёт в дочерних процессах, чтобы краш CUDA или OOM не могли уронить сервер:

- **Инференс SAM** → `backend/sam_runtime.py` (`SamRuntime`) держит один долгоживущий подпроцесс (`python -m ml_backend sam`), общающийся через **JSON-lines по stdin/stdout**. На 8 ГБ GPU помещается только одна модель, поэтому переключение между интерактивом (SAM 2) и авторазметкой (SAM 3) **перезапускает подпроцесс**, освобождая VRAM.
- **Обучение** → `backend/train_runtime.py` запускает **одноразовый** подпроцесс (`python -m ml_backend train` или `train-yolo --config <json>`) на каждый прогон. Прогресс стримится в **stdout** как JSON-lines (парсится в `status` + поэпоховую `history`), человекочитаемые логи — в **stderr** (кольцевой буфер = «живой терминал»). Остановка пишет файл-сентинел `.stop`, который тренер опрашивает. Одновременно только один прогон.
- **`ml_backend/__main__.py`** — точка входа подпроцессов с подкомандами: `sam`, `train` (RF-DETR через `training_service.py`), `train-yolo` (Ultralytics через `yolo_train_service.py`), `predict` (`predict_service.py`).

`backend/config.py::ensure_ml_backend_importable()` добавляет `QUICKLABEL_DIR` в `sys.path`, выставляет `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` и экспортирует `ML_BACKEND_SAM2_PATH`/`SAM3_PATH`, чтобы поиск весов никогда не зависел от cwd.

### Внутрипроцессные задачи (jobs)

Вызовы SAM API возвращаются сразу и выполняются на **одном рабочем потоке** (`backend/jobs.py::Job`); UI опрашивает `/api/jobs/{id}` и может отменить (`cancel`). Многокадровое распространение проверяет `is_cancelled()` между изображениями (настоящая отмена); отдельный вызов torch прервать нельзя, поэтому его отмена просто перестаёт ждать результат в UI.

### Хранение данных

`backend/store.py::ProjectStore` — один проект = одна папка `projects/<slug>/` с единственным `project.json` (классы, изображения, аннотации, `static_rois`). Загруженные изображения **копируются** в `<project>/images/`; импортированные «по папке» — **ссылаются по абсолютному пути** (поэтому при переносе на другую машину эти пути должны совпадать). У аннотаций есть `status: "suggested"` (от SAM, пунктир в UI) против подтверждённых.

**Статичные ROI** — рамки на уровне проекта в нормализованных координатах 0..1, применяемые ко *всем* кадрам (масштабируются под размер каждого изображения), со списком `exceptions: [image_id…]` для исключения отдельных кадров.

### Конвейер экспорта / импорта

- `backend/export_common.py` — общий конвейер: split train/val(/test) **на уровне исходных изображений** (аугментации только в train, чтобы избежать утечки в валидацию) + впекание статичных ROI + аугментация (повороты с точным пересчётом аннотаций).
- `backend/yolo_export.py` — layout Ultralytics (`images/{train,val}`, `labels/{train,val}`, `data.yaml`), detection или segmentation.
- `backend/coco_export.py` — COCO JSON в стиле Roboflow для RF-DETR (`_annotations.coco.json`, `bbox`+`segmentation`).
- `backend/dataset_import.py` — обратная операция: читает ранее экспортированный YOLO- или COCO-датасет обратно в проект как подтверждённые аннотации (формат определяется автоматически). Для продолжения частично размеченных датасетов.

Страница обучения переиспользует эти экспортёры для сборки датасета в `projects/<slug>/_train/<run_id>/dataset` перед запуском тренера.

### Сервер

`backend/server.py` — приложение FastAPI. Маршруты `/api/*` для проектов, классов, изображений, аннотаций, static_rois, `sam2/points`, `sam2/box`, `sam3/auto`, `sam3/propagate`, jobs, export, train (`check`/`status`/`stop`), trained_models, predict/validate. Статика `web/` монтируется **последней** на `/` (чтобы `/api` имел приоритет). `main()` поднимает uvicorn и открывает браузер.

## Соглашения

- Соблюдайте существующий стиль докстрингов по модулям: каждый файл backend начинается с абзаца, объясняющего его *роль и причины такого устройства* (особенно обоснование изоляции через подпроцессы). Сохраняйте это при правках.
- Пользовательский текст — на **русском**; новые строки UI/логов держите в согласии с окружающим кодом.
- Пути и порты переопределяются через переменные окружения: `QUICKLABEL_PROJECTS`, `QUICKLABEL_MODELS`, `QUICKLABEL_HOST`, `QUICKLABEL_PORT`, `QUICKLABEL_PYTHON`, `ML_BACKEND_SAM2_PATH`, `ML_BACKEND_SAM3_PATH`.
- dtype для SAM 3 выбирается автоматически по compute capability GPU (fp16 на Turing/до Ampere, иначе bf16) в `ml_backend/sam_service.py` — не хардкодьте его.
