import streamlit as st

from log10.load import OpenAI

import uuid
import requests
import os
import time

import pandas as pd

client = OpenAI()

import streamlit as st


def fetch_autofeedback(id):
    api_token = os.getenv("LOG10_TOKEN")
    org_id = os.getenv("LOG10_ORG_ID")

    url = "https://graphql.log10.io/graphql"
    headers = {"content-type": "application/json", "x-api-token": api_token}
    query = """
    query OrganizationCompletion($orgId: String!, $id: String!) {
        organization(id: $orgId) {
            slug
            completion(id: $id) {
                id
                autoFeedback {
                    id
                    status
                    jsonValues
                    comment
                    task {
                        id
                        name
                    }
                }
            }
        }
    }
    """
    variables = {"orgId": org_id, "id": id}
    payload = {"query": query, "variables": variables}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def poll_auto_feedback(completion_id):

    done = False
    retries = 60
    while not done:
        af = fetch_autofeedback(completion_id)
        if (
            af.get("data", {})
            .get("organization", {})
            .get("completion", {})
            .get("autoFeedback")
        ):
            return (
                af.get("data", {})
                .get("organization", {})
                .get("completion", {})
                .get("autoFeedback")
            )

        time.sleep(2)

        retries -= 1

        if retries == 0:
            raise Exception("Timeout waiting for autofeedback")


def add_completion(tags, article, summary):
    completion_id = str(uuid.uuid4())
    token = os.getenv("LOG10_TOKEN")
    org_id = os.getenv("LOG10_ORG_ID")
    url = os.getenv("LOG10_URL")
    url = f"{url}/api/v1/completions/{completion_id}"
    headers = {
        "Content-Type": "application/json",
        "X-Log10-Token": token,
        "X-Log10-Organization": org_id,
    }

    data = {
        "duration": 10,
        "id": completion_id,
        "kind": "chat",
        "organization_id": org_id,
        "request": {
            "messages": [
                {
                    "content": "you are an expert summarizer",
                    "index": 0,
                    "role": "system",
                },
                {
                    "content": "summarize the following text",
                    "index": 1,
                    "role": "user",
                },
                {"content": article, "index": 2, "role": "user"},
            ],
            "model": "test",
        },
        "stack_trace": [],
        "status": "finished",
        "tags": tags,
        "response": {
            "id": "chatcmpl-9ZPNCJCa6sy6YZFiLIgJrZ7IuIkWH",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "system_fingerprint": "foobar",
            "choices": [
                {
                    "logprobs": None,
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "content": summary,
                        "index": 3,
                        "role": "assistant",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        },
    }

    response = requests.post(url, headers=headers, json=data)
    return completion_id


def grade_cnn_row(rows):
    # Make an array copy of the rows
    rows = [row.to_dict() for _, row in rows.iterrows()]

    completion_progress = st.progress(
        0, text=f"Sending {len(rows)} articles. Please wait."
    )

    # Send all completions at once
    for i in range(len(rows)):
        row = rows[i]
        print(row)
        completion_icl = add_completion(
            ["log10/summary-grading"], row["Article"], row["Summary"]
        )

        completion_lsr = add_completion(
            ["log10/coverage-scoring"], row["Article"], row["Summary"]
        )

        completion_progress.progress(i / len(rows), text=f"Sent {i} articles")

    completion_progress.empty()

    # TODO: Poll concurrently
    feedback_progress = st.progress(0, text="Grading articles. Please wait.")
    for i in range(len(rows)):
        row = rows[i]
        start_time = time.time()
        feedback_icl = poll_auto_feedback(completion_icl)
        duration_icl = time.time() - start_time
        print(f"ICL took {duration_icl} seconds")

        start_time = time.time()
        feedback_lsr = poll_auto_feedback(completion_lsr)
        duration_lsr = time.time() - start_time
        print(f"LSR took {duration_lsr} seconds")

        print(feedback_icl)
        print(feedback_lsr)

        row["ICL Coverage"] = feedback_icl["jsonValues"]["Coverage"]
        row["LSR Coverage"] = feedback_lsr["jsonValues"]["coverage"]

        # Compute absolute error for ICL and LSR
        row["ICL Error"] = abs(row["Ground Truth Coverage"] - row["ICL Coverage"])
        row["LSR Error"] = abs(row["Ground Truth Coverage"] - row["LSR Coverage"])

        row["ICL Duration"] = duration_icl
        row["LSR Duration"] = duration_lsr

        feedback_progress.progress(i / len(rows), text=f"Graded {i} articles")

    feedback_progress.empty()

    return rows


def page1():
    st.title("CNN News dataset")

    from datasets import load_dataset

    if "ds" not in st.session_state:
        print("Resetting dataset")
        st.session_state["ds"] = load_dataset("openai/summarize_from_feedback", "axis")

    if "af_rows" not in st.session_state:
        print("Resetting af rows")
        st.session_state["af_rows"] = []

    if "num_rows" not in st.session_state:
        print("Resetting num rows")
        st.session_state["num_rows"] = 0

    # if st.button("🪄 Grade 10 more"):
    #     st.session_state["num_rows"] += st.session_state["num_rows"] + 10

    ds = st.session_state["ds"]
    test_set = ds["test"].to_pandas()

    test_set = test_set[test_set["batch"] == "cnndm3"]

    input_def = pd.DataFrame()

    input_def["Article"] = test_set["info"].apply(lambda x: x["article"])
    input_def["Summary"] = test_set["summary"].apply(lambda x: x["text"])
    input_def["Ground Truth Coverage"] = test_set["summary"].apply(
        lambda x: x["axes"]["coverage"]
    )

    # Get slice of rows to grade - provide in bulk
    rows_to_grade = input_def.iloc[
        st.session_state["num_rows"] : st.session_state["num_rows"] + 10
    ]

    rows = grade_cnn_row(rows_to_grade)

    # print(f"Num rows: {st.session_state['num_rows']}")
    # for i in range(st.session_state["num_rows"]):
    #     print(input_def.iloc[i])

    #     row = None
    #     if i < len(st.session_state["af_rows"]):
    #         row = st.session_state["af_rows"][i]
    #     else:
    #         row = grade_cnn_row(input_def.iloc[i])

    #     rows.append(row)

    output_df = pd.DataFrame(
        rows,
        columns=[
            "Article",
            "Summary",
            "Ground Truth Coverage",
            "ICL Coverage",
            "LSR Coverage",
            "ICL Error",
            "LSR Error",
        ],
    )

    st.dataframe(output_df)

    import matplotlib.pyplot as plt
    import numpy as np

    # Seed for reproducibility
    np.random.seed(42)

    # Generate the data
    # Create errors from rows (ICL error and LSR error)
    errors_ICL = output_df["ICL Error"]
    errors_LSR = output_df["LSR Error"]

    # Create a figure and a subplot for the boxplot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the boxplot
    ax.boxplot([errors_ICL, errors_LSR], labels=['ICL', 'LSR'])

    # Set title and labels
    ax.set_title('Box Plot of Absolute Errors')
    ax.set_ylabel('Absolute Error')

    # Display the plot in Streamlit
    st.pyplot(fig)
    # ax.hist(arr, bins=20)


def sandbox():
    st.title("Sandbox")

    if "summary" not in st.session_state:
        st.session_state["summary"] = (
            """Charity runners taking part in a 10k event in Bournemouth Bay were sent on an unscheduled two-mile detour, leaving them exhausted and completely dehydrated."""
        )

    if "article" not in st.session_state:
        st.session_state["article"] = (
            "Charity runners taking part in a 10km fun run at the weekend were left exhausted after being sent on an unscheduled two-mile detour. The blunder was believed to have been caused by a race marshal taking a toilet break during the event, missing 300 runners who should have been directed at a junction point. Instead they continued past the unmanned marshall point and had to run for an extra three kilometres while the other 900 competitors followed the correct route. Scroll down for video Blunder: Charity runners taking part in yesterday's Bournemouth Bay 10K Run (pictured) were left exhausted after being sent on an unscheduled two-mile detour The bizarre gaffe happened during yesterday's Bournemouth Bay Run and today the organisers - Bournemouth Borough Council - appealed for those who were affected by the mix-up to contact them for a 'gesture of goodwill.' A local authority spokesman said that it was investigating what happened to the marshal who should have directed runners at a turning point. It was reported that some runners were 'in tears' while one described the event's organisation as 'shambolic'. Hayley James, who is four months pregnant and from Poole, said: 'To have a race of that scale with only one marshal on a point is inexcusable. 'We saw loads of people walking at the end, some were in tears, I felt so sorry for them - I felt like crying at the 10km mark.' Andy Isaac, from Bournemouth, said the event was 'mayhem' with one point where an elderly woman managed to drive onto the route and was flashing her lights at oncoming runners. A map shows where up to 300 runners continued along the coastal path after a marshal who was meant to direct them on to a turn went to the toilet Reaction: Two people vent their frustration at the Bournemouth Bay Run on Twitter yesterday It also emerged that water stations ran out of supplies during the race, forcing some runners to drink from half-empty bottles that had been left on the ground as they battled against dehydration. Commenting on the Daily Echo website, one runner said: 'We had a bottle of water at the three mile station, but at the six mile point they had totally ran out, so nothing'. Jon Weaver, head of resort marketing and events at the council, said: 'Unfortunately there was some confusion with marshalling arrangements at one point, but it was a critical point. We apologise unreservedly to those front runners. 'In 33 years of running... this is the first time this has happened and as part of our debrief we will be analysing the arrangements carefully... to learn for 2016. 'We understand runners have trained for a long time for the event and it's hard for them and we do empathise with how they are feeling.' It was hoped that the event would have raised more than \u00a370,000 for the British Heart Foundation. Some racers took to Twitter to vent their frustration over the blunder. Rob Kelly wrote: Really disappointed in the #BournemouthBayRun 10k that ended up 13k very poor show bad marshalling #wontbeback.' AndKim Kelly replied: 'Totally agree and never got to do a 5k as they were 45mins behind schedule :(((.",
        )

    st.session_state["article"] = st.text_area(
        "Source text",
        st.session_state["article"],
        height=600,
    )

    model = st.selectbox(
        "Summarizer model",
        ("gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"),
    )

    if st.button("Summarize"):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize the following text:"},
                {"role": "user", "content": st.session_state["article"]},
            ],
        )

        st.session_state["summary"] = response.choices[0].message.content

    st.session_state["summary"] = st.text_area(
        "Summary", st.session_state["summary"], height=200
    )

    if st.button("🪄 Grade"):
        # TODO: Send manually crafted log to log1 twice (one for standard ICL summarizer, and one for LSR summarizer)
        completion_1 = add_completion(
            ["log10/summary-grading"],
            st.session_state["article"],
            st.session_state["summary"],
        )
        completion_2 = add_completion(
            ["log10/coverage-scoring"],
            st.session_state["article"],
            st.session_state["summary"],
        )

        # Make progress bar
        progress_bar = st.progress(0.25, text="Waiting for autofeedback")

        feedback_1 = poll_auto_feedback(completion_1)
        progress_bar.progress(0.5, text="Received feedback for ICL")

        feedback_2 = poll_auto_feedback(completion_2)
        progress_bar.progress(1.0, text="Received feedback for LSR")

        progress_bar.empty()

        col1, col2 = st.columns(2)

        print(feedback_2)

        icl_coverage = feedback_1.get("jsonValues", {}).get("Coverage", 0)
        lsr_coverage = feedback_2.get("jsonValues", {}).get("coverage", 0)

        col1.progress(icl_coverage / 7, text=f"ICL Coverage: {icl_coverage}")
        col2.progress(lsr_coverage / 7, text=f"LSR Coverage: {lsr_coverage}")


# import pandas as pd
# import numpy as np

# df = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))

# st.dataframe(df)


pg = st.navigation(
    {
        "AutoFeedback": [
            st.Page(page1, title="CNN News dataset", icon=":material/newsmode:"),
            st.Page(sandbox, title="Sandbox", icon=":material/play_arrow:"),
        ]
    }
)
pg.run()
