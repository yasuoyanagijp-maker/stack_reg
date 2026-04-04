# ARIAKE OCTA Stack Registration Tool

Retinal OCTA 画像スタックのレジストレーションと平均化のためのデスクトップアプリです。ARIAKE ImageJ マクロの Python 実装をベースに、単体実行と操作性を重視しています。

## 機能

- **ImageJ との整合**: 前処理、4 倍拡大、CLAHE 最適化などを ImageJ の挙動に合わせて再現。
- **参照ベースのアライメント**: 「平均スタック（画像 5）」方式による Affine レジストレーション。
- **データ検証**: 患者フォルダ間でファイル順序と整合性を自動確認。
- **UI**: Flet によるモダンなクロスプラットフォーム UI。

## 要件

- **Python 3.14**（推奨。プロジェクトは 3.14 前提で開発）
- 依存関係: `requirements.txt`（現行 venv に合わせ `pip freeze` でバージョン固定）

仮想環境（`venv/`、`.venv/` など）とキャッシュ類は `.gitignore` で除外されています。

## インストール

1. **リポジトリを取得**:
   ```bash
   git clone <repository-url>
   cd stack_reg
   ```

2. **仮想環境**:
   ```bash
   python3.14 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **依存関係のインストール**:
   ```bash
   pip install -r requirements.txt
   ```

## 実行

開発モードでアプリを起動:

```bash
python -m app.main
```

## デスクトップ向けパッケージ（.exe / .dmg）

`flet build` でネイティブ実行ファイルを作成できます。

### Windows (.exe)

```bash
flet build windows
```

### macOS (.dmg)

```bash
flet build macos
```

成果物は `build/` 配下に出力されます（`build/` は Git 対象外）。

## プロジェクト構成（概要）

| パス | 内容 |
|------|------|
| `app/main.py` | エントリ（`ft.run`） |
| `app/ui/` | ダッシュボード・QC ビューアなど |
| `app/core/` | 画像処理・レジストレーション・パイプライン |
| `archive/` | 開発・検証用スクリプト（アプリ本体からは未使用） |

## 開発メモ

- Flet は **v0.84+** を想定（`FilePicker` は `page.services` に追加するなど API 変更あり）。
- UI / API の細則は `.cursorrules` を参照。

---

Developed by Team Yanagi (2025/2026)
