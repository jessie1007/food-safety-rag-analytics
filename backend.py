import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import plotly.express as px
import duckdb

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
con = duckdb.connect(str(DATA_DIR / "food_inspections_clean.duckdb"), read_only=True)

load_dotenv()
#embeded text file
rag_df = pd.read_pickle(DATA_DIR / "rag_df.pkl")
index = faiss.read_index(str(DATA_DIR / "faiss_index.index"))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)


rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
            You are a food safety inspection analyst.
            Use only the retrieved inspection records.
            Do not add causes, recommendations, or facts unless they are directly supported by the retrieved records.
            If the records do not provide enough evidence, say so.
            Keep the answer concise and evidence-based.
            """),
    ("human", """
            User question:
            {question}

            Retrieved inspection records:
            {context}

            Answer using this format:

            1. Common pattern:
            2. Evidence from retrieved records:
            3. Practical corrective actions supported by the records:
            4. Limitations:
""")
])

rag_chain = rag_prompt | llm | StrOutputParser()

sql_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a food safety data analyst.

Use only the provided SQL result.
Do not invent numbers,names, years, causes, or trends.
If the SQL result is limited, say so.
Keep the answer concise and business-oriented.
If question is complicated, you can use multiple SQL tools to answer the question.
Keep the answer concise.
Use 2–4 short bullet points maximum.
Do not restate every monthly value unless specifically requested.
"""),
    ("human", """
        User question:
        {question}

        SQL tool used:
        {tool_name}

        SQL result:
        {result}

Write the answer using this format:

1. Direct answer:
2. Key evidence from SQL result:
3. Business interpretation:
4. Limitation:
""")
])

sql_summary_chain = sql_summary_prompt | llm | StrOutputParser()

# LLM router template to determine which tool to use
router_template = """
You are a routing assistant for a food safety analytics system.

Choose exactly one tool:

top_violation_types
- Use for questions about the most common violation types.

violation_trend_by_month
- Use for questions about violation trends over time.

violation_trend_by_severity
- Use for questions about trends split by severity.

top_violation_owner
- Use for questions about repeat offenders or businesses with many serious outcomes.

similar_violations
- Use for semantic search questions asking for similar violations, related issues, or examples.

User question:
{text}

Return only the tool name.
"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["text"]
)

router_chain = router_prompt | llm | StrOutputParser()

def route_question(question):
    route = router_chain.invoke({"text": question})
    return route

# SQL functions 1 - top violation types,What violations are most common?
def top_violation_types(year=2025, severity_level=None, limit=15):
    query = """
    SELECT 
        violdesc,
        COUNT(*) AS violation_count
    FROM inspections
    WHERE violdesc IS NOT NULL 
      AND violdesc <> 'Unknown violation'
      AND severity_score IS NOT NULL
      AND (? IS NULL OR EXTRACT(year FROM resultdttm) = ?)
      AND (? IS NULL OR severity_level = ?)
    GROUP BY violdesc
    ORDER BY violation_count DESC
    LIMIT ?
    """
    return con.execute(
        query,
        [year, year, severity_level, severity_level, limit]
    ).df()

# SQL functions 2 - violation trend by month
def violation_trend_by_month(start_year=2021):
    query = """
    SELECT
        month,
        COUNT(*) AS violation_count
    FROM inspections
    WHERE month IS NOT NULL
      AND violdesc IS NOT NULL
      AND violdesc <> 'Unknown violation'
      AND severity_score IS NOT NULL
      AND year >= ?
    GROUP BY month
    ORDER BY month
    """
    return con.execute(query, [start_year]).df()

def violation_trend_by_month_and_severity(start_year=2021):
    query = """
    SELECT
        month,
        severity_level,
        COUNT(*) AS violation_count
    FROM inspections
    WHERE month IS NOT NULL
      AND violdesc IS NOT NULL
      AND violdesc <> 'Unknown violation'
      AND severity_score IS NOT NULL
      AND severity_level IS NOT NULL
      AND year >= ?
    GROUP BY month, severity_level
    ORDER BY month, severity_level
    """
    return con.execute(query, [start_year]).df()

def severity_by_zip(limit=15):
    query = """
    SELECT
        zip,
        COUNT(*) AS total_violations,
        AVG(severity_score) AS avg_severity,
        SUM(CASE WHEN severity_level = 'high' THEN 1 ELSE 0 END) AS high_severity_count
    FROM inspections
    WHERE zip IS NOT NULL
      AND severity_score IS NOT NULL
    GROUP BY zip
    ORDER BY high_severity_count DESC, avg_severity DESC
    LIMIT ?
    """
    return con.execute(query, [limit]).df()

def top_violation_owner(year=2025, limit=15):
    query = """
    SELECT
        businessname,
        address,
        COUNT(*) AS total_violations,
        SUM(CASE WHEN result_group IN (
            'fail',
            'extended_fail',
            'hearing',
            'temporary_suspension',
            'voluntary_closure_avoid',
            'forced_closure'
        ) THEN 1 ELSE 0 END) AS serious_outcome_count,
        AVG(severity_score) AS avg_severity
    FROM inspections
    WHERE businessname IS NOT NULL
      AND severity_score IS NOT NULL
      AND (? IS NULL OR EXTRACT(year FROM resultdttm) = ?)
    GROUP BY businessname, address
    HAVING COUNT(*) > 10
    ORDER BY serious_outcome_count DESC, avg_severity DESC
    LIMIT ?
    """
    return con.execute(query, [year, year, limit]).df()

# chart 1, monthly violation trend
def chart_monthly_violation_trend(df):
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])

    fig = px.scatter(
        df,
        x="month",
        y="violation_count",
        trendline="ols",
        title="Food Safety Violations Trend"
    )

    fig.add_scatter(
        x=df["month"],
        y=df["violation_count"],
        mode="lines+markers",
        name="Monthly Count",
        #line=dict(color="blue", width=3),
        #marker=dict(color="blue")
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        title_x=0.5,
        xaxis_title="Month",
        yaxis_title="Violation Count"
    )

    return fig



def chart_violation_trend_by_severity(df):
    df = df.copy()

    colors = {
        "low": "green",
        "medium": "blue",
        "high": "red"
    }

    fig = px.line(
        df,
        x="month",
        y="violation_count",
        color="severity_level",
        color_discrete_map=colors,
        markers=True,
        title="Monthly Food Safety Recorded Violations by Severity Trend"
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        title_x=0.5,
        xaxis_title="Month",
        yaxis_title="Violation Count"
    )

    return fig

def chart_top_violation_types(df):
    df = df.copy()
    fig = px.bar(
        df.sort_values("violation_count"),
        x="violation_count",
        y="violdesc",
        orientation="h",
        title="Top Food Inspection Violation Types"
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        title_x=0.5,
        xaxis_title="Violation Count",
        yaxis_title="Violation Type"
    )

    fig.show()

def chart_top_violation_owner(df, year=None):
    df = df.copy()

    title_year = f" in {year}" if year else " Overall"
    fig = px.bar(
        df.sort_values("serious_outcome_count"),
        x="serious_outcome_count",
        y="businessname",
        orientation="h",
        title=f"Top violation business{title_year}",
        hover_data={
            "total_violations": True,
            "avg_severity": ":.2f",
            "businessname": False
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        title_x=0.5,
        xaxis_title="Serious Outcome Count",
        yaxis_title="Business"
    )

    return fig

def retrieve_similar_violations(query, k=3):
    #convert query to same embedding
    query_embedding = embed_model.encode(
        #embeds query to a list as faiss expect 2D array
        [query],
        normalize_embeddings=True
    ).astype("float32")
    #search for top k similar violations and store scores and indices
    scores, indices = index.search(query_embedding, k)
    #get the results
    results = rag_df.iloc[indices[0]].copy()
    results["similarity_score"] = scores[0]

    return results[
        [
            "similarity_score",
            "businessname",
            "address",
            "violdesc",
            "severity_level",
            "result",
            "inferreddescription",
            "comments"
        ]
    ]

def format_rag_context(results, max_rows=5):
    rows = []

    for i, row in results.head(max_rows).iterrows():
        rows.append(f"""
            Record {i+1}
            Business: {row.get("businessname")}
            Violation: {row.get("violdesc")}
            Severity: {row.get("severity_level")}
            Inspection Result: {row.get("inferreddescription")}
            Comment: {row.get("comments")}
            """.strip())

    return "\n\n".join(rows)

def summarize_sql(question, route, df, max_rows=10):
    result_text = df.head(max_rows).to_string(index=False)

    summary = sql_summary_chain.invoke({
        "question": question,
        "tool_name": route,
        "result": result_text
    })

    return summary

def summarize_rag(question, retrieved_df):
    context = format_rag_context(retrieved_df)

    summary = rag_chain.invoke({
        "question": question,
        "context": context
    })

    return summary

#build function to execute the route and return the anwser
def ask(question, year=None, limit=10, k=3, show_chart=True):
    #route to route_question tool to determine which tool to use, output tool name
    route = route_question(question).strip()

    #if tool is top_violation_types, run top_violation_types sql tool
    if route == "top_violation_types":
        df = top_violation_types(year=year, limit=limit)
        #if show_chart is true, show chart
        fig = chart_top_violation_types(df) if show_chart else None
        summary = summarize_sql(question, route, df)

        #return tool name, type of the tool (sql or rag), data details, summary of the result
        return {
            "route": route,
            "type": "sql",
            "data": df,
            "summary": summary
        }
    #if tool is top_violation_owner, run top_violation_owner sql tool
    elif route == "top_violation_owner":
        df = top_violation_owner(year=year, limit=limit)
        #if show_chart is true, show chart
        fig = chart_top_violation_owner(df, year=year) if show_chart else None
        #use sql_summary function to summarize the result
        summary = summarize_sql(question, route, df)

        #return tool name, type of the tool (sql or rag), data details, summary of the result
        return {
            "route": route,
            "type": "sql",
            "data": df,
            "summary": summary
        }
    elif route == "violation_trend_by_month":
        df = violation_trend_by_month(start_year=year or 2021)
        fig = chart_monthly_violation_trend(df) if show_chart else None
        summary = summarize_sql(question, route, df)
        return {
            "route": route,
            "type": "sql",
            "data": df,
            "summary": summary
        }

    elif route == "violation_trend_by_severity":
        df = violation_trend_by_month_and_severity(start_year=year or 2021)
        fig = chart_violation_trend_by_severity(df) if show_chart else None
        summary = summarize_sql(question, route, df)
        return {
            "route": route,
            "type": "sql",
            "data": df,
            "summary": summary
        }
    #if tool is similar_violations, run similar_violations rag tool
    elif route == "similar_violations":
        df = retrieve_similar_violations(question, k=k)
        #use summarize_rag function to summarize the result
        summary = summarize_rag(question, df)

        return {
            "route": route,
            "type": "rag",
            "data": df,
            "summary": summary
        }
    #if tool is not found, return error
    else:
        return {
            "route": route,
            "type": "error",
            "data": None,
            "summary": "No matching tool found."
        }