# Third-party notices / 第三者ライセンス

MidnaUdon EventPhotoSorterの独自ソースコードと文書は、ルートの `LICENSE`（MIT）に従います。以下の第三者コンポーネントをMITへ変更するものではありません。それぞれの著作権表示・ライセンスを保持してください。ブランドロゴは [ASSET_NOTICE.md](ASSET_NOTICE.md) を参照してください。

## Pythonライブラリ・実行環境

ライセンス原文とバージョン一覧は [licenses/INDEX.md](licenses/INDEX.md)、機械可読の一覧は [licenses/inventory.json](licenses/inventory.json) です。原文はビルドに使用したインストール済みパッケージから収集しています。ビルド用・任意機能用の未使用コンポーネントを含む場合があります。

主なライブラリはOpenAI Python SDK、Pillow、NumPy、Pydantic、python-dotenv、PyTorch、torchvision、Transformers、Hugging Face Hub、Tokenizers、Safetensorsです。間接的な依存ライブラリおよびPyTorch/Pillowなどに含まれる第三者コードの通知も同梱しています。

- Pythonの原文：[licenses/runtime/PYTHON-LICENSE.txt](licenses/runtime/PYTHON-LICENSE.txt)。Windowsランタイムの追加条件を含みます。
- Tcl/Tkの原文：`licenses/runtime/tcl-tk/`。
- Microsoft Visual C++ランタイムはVisual Studio 2022の再配布用CRTから、バイナリを改変せず同梱します。利用・再配布の条件は [RUNTIME_TERMS.txt](RUNTIME_TERMS.txt)、原文は [licenses/runtime/msvc/VS2022-LICENSE.txt](licenses/runtime/msvc/VS2022-LICENSE.txt) です。これらのMicrosoft部品にアプリ本体のMITライセンスを適用するものではありません。
- CPU版の実際のネイティブ部品・取得元・SHA-256は、ビルド時に生成される `licenses/native_dependencies.json` に記録します。Windows 10/11標準のDbgHelp・DbgCore・UCRT・API-set DLLは同梱せず、OSの部品を利用します。NumPyが要求する別名のMSVCP DLLも、同じ公式再配布用CRTの内容を保持しています。
- PyInstallerはGPLと配布用例外条項を持ちます。アプリのライセンスは同じGPLにする必要はありません。[公式説明](https://pyinstaller.org/en/stable/license.html)
- certifiのCA証明書データにはMPL-2.0が適用されます。配布したものと同じソース形式の `cacert.pem` を `licenses/python/certifi/source/` に同梱しています。これは公開CA証明書で、個人のAPIキーや秘密鍵ではありません。
- tqdmにはMPL-2.0とMITが適用されます。ビルドで使用したものと同じ編集可能なソースを `licenses/python/tqdm/source/` に同梱しています。MPL全文は `licenses/upstream/mpl-2.0/LICENSE` にあります。
- OpenSSL 3の原文：`licenses/upstream/openssl/`。画像コーデック等の補足原文：`licenses/upstream/libjpeg-turbo/`、`libpng/`、`libwebp/`、`zlib/`。このソフトウェアの一部はIndependent JPEG Groupの成果に基づきます。
- Pillowに含まれるFreeTypeはFreeType License（FTL）の条件で利用しています。This software is based in part on the work of the FreeType Team. [FreeType Project](https://freetype.org/) / 原文は `licenses/python/pillow/` のLICENSEに含まれます。
- GPU版を別途ビルドした場合は、GPU環境で `tools/collect_licenses.py` を実行して通知を追加してください。CUDA関連の第三者条件もその配布物に従います。

## 機械学習モデル

| モデル | 提供元 | ライセンス | 原文 |
| --- | --- | --- | --- |
| [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) | Meta / Facebook Research（DINOv2） | Apache-2.0 | [LICENSE](licenses/upstream/dinov2/LICENSE) |
| [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | IDEA Research（Grounding DINO） | Apache-2.0 | [LICENSE](licenses/upstream/grounding-dino/LICENSE) |

モデルの重みを独自に改変・再学習していません。ソース公開用ZIPには重みを含めず、実行時または `tools/download_models.py` で取得します。通常のEXE配布には取得済みの重みを同梱します。DINOv2の別モデル（X-Ray/Cell DINO等）のライセンスとは区別してください。

## 原文の取得と再生成

`tools/collect_licenses.py` は現在のビルド環境から原文を収集します。補足原文は `tools/fetch_upstream_licenses.py` で取得でき、URLとSHA-256は `licenses/upstream/sources.json` にあります。

依存関係やモデルを変更して再配布する場合は、変更後の通知を収集して一緒に配布してください。OpenAI APIは外部サービスであり、APIの利用料金・サービス条件はこのソフトウェアのMITライセンスとは別です。
