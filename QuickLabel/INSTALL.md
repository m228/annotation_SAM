# QuickLabel — Инструкция по установке и обновлению

## Требования

| | |
|---|---|
| ОС | Windows 10/11 (64-bit) |
| Python | 3.13 ([python.org](https://www.python.org/downloads/)) |
| GPU | NVIDIA с CUDA-совместимыми драйверами (опционально; без GPU — CPU-режим, SAM 3 будет медленным) |
| Место на диске | ~8 ГБ (venv + модели) |

---

## Первая установка

### Шаг 1 — Получить файлы релиза

Скачайте последний `QuickLabel_vX.Y.Z.zip` с
[GitHub Releases](https://github.com/m228/annotation_SAM/releases/latest)
и распакуйте в любую папку, например `C:\QuickLabel\`.

После распаковки структура:
```
C:\QuickLabel\
  QuickLabel\
    QuickLabel.exe
    setup.ps1
    run.ps1
    run.bat
    update.ps1
    backend\
    ml_backend\
    web\
    ...
```

### Шаг 2 — Добавить модели и wheels (делается один раз)

Релизный zip **не содержит** тяжёлые файлы (модели SAM ~4.3 ГБ).
Скопируйте их вручную или запросите у автора:

```
QuickLabel\
  models\
    sam2.1_hiera_large.pt   (~4 ГБ)
    sam3.pt                 (~300 МБ)
  wheels\
    sam2-1.1.0-py3-none-any.whl
    sam3-0.1.0-py3-none-any.whl
```

### Шаг 3 — Создать окружение Python

Откройте PowerShell **в папке `QuickLabel\`** и выполните:

```powershell
# GPU-ускорение (NVIDIA, CUDA 12.4) — рекомендуется
.\setup.ps1

# Если нет NVIDIA GPU:
.\setup.ps1 -CpuOnly
```

Скрипт создаст `.venv\` и установит все зависимости (займёт 5–15 мин).

### Шаг 4 — Запуск

Двойной клик по **`QuickLabel.exe`** — откроется консоль сервера
и через ~2 с браузер на `http://127.0.0.1:8765`.

Чтобы остановить — закройте консольное окно.

---

## Обновление до новой версии

### Автоматически (рекомендуется)

Запустите **`update.bat`** (или `update.ps1`) из папки `QuickLabel\`.
Скрипт скачает последний релиз, распакует поверх и сохранит:
- `.venv\` — окружение Python (не пересоздаётся)
- `models\` — веса моделей
- `wheels\` — SAM-колёса
- `projects\` — ваши данные разметки

### Вручную

1. Скачайте новый `QuickLabel_vX.Y.Z.zip`.
2. Распакуйте поверх текущей папки `QuickLabel\`
   (`.venv`, `models`, `wheels`, `projects` — не трогайте).
3. Готово.

---

## Перенос на другой компьютер

Скопируйте папку `QuickLabel\` **без `.venv\`** (venv непереносим):
```
QuickLabel\       <- копировать целиком
  models\         <- внутри, ~4.3 ГБ
  wheels\         <- внутри
  projects\       <- ваши данные (опционально)
  .venv\          <- НЕ копировать
```

На новом ПК: установить Python 3.13, затем `.\setup.ps1` из папки `QuickLabel\`.

---

## Сборка exe из исходников (для разработчиков)

```powershell
cd QuickLabel
.\build_exe.ps1
# -> QuickLabel.exe (в папке QuickLabel)
# -> dist\QuickLabel_vX.Y.Z.zip (готов для релиза)
```

Публикация релиза вручную:
```powershell
gh release create vX.Y.Z "dist\QuickLabel_vX.Y.Z.zip" -t vX.Y.Z --generate-notes
```

Автоматический релиз через GitHub Actions — достаточно создать тег:
```powershell
git tag v1.0.1
git push origin v1.0.1
# Actions сам соберёт exe и создаст Release
```

---

## Устранение неполадок

| Проблема | Решение |
|---|---|
| `Python venv not found` | Запустите `.\setup.ps1` из папки `QuickLabel\` |
| Браузер не открылся | Зайдите вручную на `http://127.0.0.1:8765` |
| SAM 3 медленный (минуты) | Нет NVIDIA GPU или драйверы устарели. Пересоберите с `-CpuOnly` — не поможет, но хотя бы не зависнет |
| `Model not found` | Проверьте что `models\sam2.1_hiera_large.pt` и `models\sam3.pt` на месте |
| Ошибка при `update.bat` | Нет интернета или GitHub недоступен. Обновитесь вручную |
