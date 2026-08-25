# Agentic-Router 實驗進度與成果報告

## 1. 專案概述 (Project Overview)
**Agentic-Router** 是一個免訓練 (Training-free) 的大型語言模型 (LLMs) 路由框架，旨在透過測試階段的「查詢難度推理」與「模型能力剖析」，將使用者的 Query 自動分配給最適合且最具成本效益的 LLM。

相較於現有的透過監督式學習 (如 RouteLLM) 或是強化學習訓練的路由模型，Agentic-Router 透過 Prompt 的方式，直接讓輕量級的大型語言模型 (SLM) 在推理階段扮演「代理人 (Agents)」的角色來做出路由決策。

### 核心方法與架構
框架由兩個主要的 Agent 構成，依賴事前建立的分析資料庫：
1. **Offline Response Analysis DB (離線回答分析庫)：** 事前針對一部分驗證集，利用評估模型 (Evaluator LLM) 分析各目標模型 (如 Llama、Mistral、GPT-4 等) 的回答表現。分析維度包含：推理 (reasoning)、理解 (comprehension)、遵循指示 (instruction following)、代理 (agentic)、知識檢索 (knowledge retrieval)、程式撰寫 (coding) 及多語言 (multilingual) 共七大向度。這些分析會被轉化為 Embedding 存入 FAISS 資料庫。
2. **Difficulty Analyst (難度分析 Agent)：** 在測試階段接收使用者 Query，並根據上述七大向度，分析該 Query 的難度與解決該 Query 所需的能力。
3. **Routing Decision Maker (路由決策 Agent)：** 透過難度分析的結果，從 FAISS 資料庫中檢索出 Top-k 相關的歷史分析紀錄。接著綜合「Query 的難度」以及擷取出的「模型過往能力表現」，推論出最適合解答此 Query 的模型。系統中會針對模型名稱進行匿名處理 (如 Model-A, Model-B) 以避免先驗偏見。

---

## 2. 目前已完成的研究成果 (Current Achievements)

1. **表現超越既有基準 (State-of-the-art Performance)：**
   - 實驗證實 Agentic-Router 在分數 (Score) 及考量成本的獎勵函數 (Rewards) 上，皆穩定超越傳統全部將任務派發給單一強模型 (如 GPT-4, Claude-v1) 的策略，並顯著優於 RouteLLM。
   - 以 `AR-Qwen` 作為 Agent 為例，相比 RouteLLM 在平均分數上提升約 33%，且在獎勵函數表現上提升 19%。

2. **驗證了 SLM 作為路由決策者的有效性：**
   - 研究發現如 Qwen (AR-Qwen) 這樣適合進行精細邏輯推理的模型，扮演路由決策 Agent 時，其表現甚至能優於一般的大型商業模型 (如 GPT-4, Gemini)。
   - AR-Qwen 的路由頻率與目標模型的實際能力呈現高度正相關 (Pearson Correlation 超過 0.75)，能在多個 Benchmark 子類別中成功識別高能力的模型。

3. **分析了引入成本 (Cost) 資訊的影響：**
   - 實驗探討若直接在 Prompt 中告知路由決策 Agent 各候選模型的成本，會導致包含 GPT-4 在內的部分 Agent 產生「Reward Hacking」的偏見，選擇極度偏向低成本模型（例如 mistral-7B），從而造成路由準確度與整體表現嚴重下降。
   - 現行版本的 Agentic-Router 在**不直接提供成本資訊**的情況下，仍能達到出色的表現與成本平衡，這歸功於它能準確匹配「Query 所需能力」與「模型實力」，避免「殺雞用牛刀」的越級指派。

4. **驗證了基於檢索的動態表現庫優勢：**
   - 實驗證明，將模型分散的歷史表現摘要為「單一靜態描述 (Static Descriptions)」會造成嚴重的資訊流失，路由效果顯著遜於 Agentic-Router 的「動態檢索 (Dynamic Retrieval)」設計。

---

## 3. 程式執行方式與技術細節 (Execution Guide)

目前 Agentic-Router 的主要實作對應到程式碼中的 **`agenticrouter_normalizedcost.py`**。這個檔案封裝了難度分析 Agent與路由決策 Agent，並串接 FAISS 向量檢索庫。

### 核心檔案與元件
- **主程式:** `agenticrouter_normalizedcost.py`
- **Embedding 模型:** `Qwen/Qwen3-Embedding-0.6B` (用於建立 FAISS 索引)
- **資料儲存 (自動建立與讀取):** 預設讀取 `faiss_difficulty_db` 和 `faiss_response_dbs` 等目錄下的 FAISS 索引檔案。
- **輸入資料:** 讀取 `router_bench_with_keywords.csv` 與包含難度分析的 `sampled_router_bench_with_difficulty_analysis.csv` 。

### 執行環境準備
系統執行前會讀取環境變數中的 API Keys。請確保有建立 `.env` 檔案並填寫：
- `OPENAI_API_KEY1` (供 GPT 系列使用)
- `GEMINI_API_KEY` (供 Gemini 使用)
*(Qwen 系列預設透過本地端 Base URL `http://0.0.0.0:8000/v1` 執行)*

### 執行命令與參數
您可以透過下令執行實驗：

```bash
python agenticrouter_normalizedcost.py [參數選項]
```

**支援的參數：**
| 參數名稱 | 型別 / 預設值 | 說明 |
| :--- | :--- | :--- |
| `--model-config` | `string` (預設: `qwen2.5`) | 指定負責扮演 Agent 的核心語言模型。支援: `qwen2.5`, `gpt-4o-mini`, `gemini-2.5-flash-lite` |
| `--routing-method` | `string` (預設: `agentic`) | 路由的策略選擇。可設定為本研究提出的 `agentic` 或者是作為比較基準的 `routellm` |
| `--ood` | `flag` | 是否啟用 Out-Of-Distribution (OOD) 測試模式。啟動後程式會使用 dataset 新的分類切分方式，並把對應 DB 路徑指向含有 `_ood` 的目錄中 |
| `--order-llm-list` | `bool` (預設: `True`) | 決定是否根據檢索所得之 scores 來排序送給提示詞中的 LLM 清單 |
| `--reverse-llm` | `bool` (預設: `True`) | 是否要在送出前反轉 LLM 清單排序 |

**執行範例：**
測試 Qwen 作為 Agent:
```bash
python agenticrouter_normalizedcost.py --model-config qwen2.5 --routing-method agentic
```

測試 Out-Of-Distribution 表現:
```bash
python agenticrouter_normalizedcost.py --model-config qwen2.5 --routing-method agentic --ood
```

測試基準方法 RouteLLM:
*(注意：需要事先跑過 `python routellm.py --mode train` 進行訓練)*
```bash
python agenticrouter_normalizedcost.py --routing-method routellm
```

---

## 4. 後續實驗規劃建議 (Future Work & Next Steps)

根據你的報告，以下是可以排入後續實驗的規劃：

1. **Agent Cost Debias (消偏機制探討)：** 既然報告中提到 Agent 在獲知 cost 後會有「Reward Hacking」(對低成本模型的過度偏好) 的現象，可以設計並驗證不同的 Prompt 技術或機制，引導 Agent 將 cost 視為參考的一環而不會過度依賴。
2. **多樣化資料集擴充 (OOD Testing)：** 目前 `agenticrouter_normalizedcost.py` 中已經實作了 `--ood` 功能，可以專注投入產出不同領域資料集下的 Out-Of-Distribution 測量與比較。
3. **Difficulty Analyst 強度擴充：** 針對簡單的問題，測試是否能由 Difficulty Analyst Agent 在發現難度極低時，直接自行生成答案返回，繞過後續的路由檢索步驟，從而進一步節省運算成本與時間。
4. **Agentic Reasoning 模型增強 (Advanced Reasoning LLM as Agent)：** 在「Predict Correctness」試驗中發現 `AR-Gemini` 識別模型對錯的能力特別強。可以嘗試將 Gemini (或其他理解力強的模型) 結合進「精準判斷能力」模組中，與善於匹配路由的 Qwen 取長補短。
5. **Reward-based Routing 分析：** 報告中提到根據經驗可直接計算預期報酬 (Reward)，並且實踐上獲得很好的成效。可以進一步實驗把此方法結合成混合型的路由決策機制 (Hybrid routing)，為某些確定性高的問題提供捷徑決策。
