import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


st.title("🏸 AI Badminton Coaching System")
# -----------------------
# EQUIPMENT DATABASE
# -----------------------

rackets = pd.DataFrame({
    "Brand": ["Yonex", "Yonex", "Li-Ning", "Li-Ning", "Victor"],
    "Model": ["Nanoflare 700", "Astrox 88D", "Turbo X90", "3D Calibar 900", "Thruster K 9900"],
    "Type": ["Head-Light", "Head-Heavy", "Balanced", "Head-Heavy", "Head-Heavy"],
    "Skill": ["Beginner", "Advanced", "Intermediate", "Advanced", "Advanced"]
})

shoes = pd.DataFrame({
    "Brand": ["Yonex", "Yonex", "Li-Ning", "Victor", "Asics"],
    "Model": ["Power Cushion 36", "65Z3", "Ranger Lite", "A960", "Gel-Blade 8"],
    "Cushion": ["Medium", "High", "Medium", "High", "High"],
    "Skill": ["Beginner", "Advanced", "Intermediate", "Advanced", "Advanced"]
})
# -----------------------
# PLAYER DATABASE FILE
# -----------------------
db_file = "players_database.csv"

if not os.path.exists(db_file):
    pd.DataFrame(columns=[
        "Fitness", "Footwork", "Smash", "Reaction",
        "Skill_Level", "Racket", "Shoes", "Months_To_Next_Level"
    ]).to_csv(db_file, index=False)


# -----------------------
# SAMPLE TRAINING DATA
# -----------------------
data = {
    "Fitness": [6,8,7,5,9,6,8,9,7,10],
    "Footwork": [5,8,7,4,9,6,8,9,7,9],
    "Smash": [4,8,7,5,9,6,8,9,6,10],
    "Reaction": [6,8,7,5,9,6,8,9,7,10],
    "Skill": [0,2,1,0,2,0,1,2,1,2]
}

df = pd.DataFrame(data)

X = df[["Fitness","Footwork","Smash","Reaction"]]
y = df["Skill"]

model = RandomForestClassifier()
model.fit(X,y)
# -----------------------
# GROWTH PREDICTION DATA
# -----------------------
growth_data = pd.DataFrame({
    "Fitness":[6,8,7,5,9,6,8,9],
    "Footwork":[5,8,7,4,9,6,8,9],
    "Smash":[4,8,7,5,9,6,8,9],
    "Reaction":[6,8,7,5,9,6,8,9],
    "Months_To_Next_Level":[8,4,5,9,3,7,4,3]
})

Xg = growth_data[["Fitness","Footwork","Smash","Reaction"]]
yg = growth_data["Months_To_Next_Level"]

growth_model = LinearRegression()
growth_model.fit(Xg, yg)


# -----------------------
# USER INPUT
# -----------------------
st.header("Enter Player Test Results")

fitness = st.slider("Fitness Level",1,10)
footwork = st.slider("Footwork Speed",1,10)
smash = st.slider("Smash Power",1,10)
reaction = st.slider("Reaction Speed",1,10)

if st.button("Analyze Player"):

    new_player = pd.DataFrame({
        "Fitness":[fitness],
        "Footwork":[footwork],
        "Smash":[smash],
        "Reaction":[reaction]
    })

    pred = model.predict(new_player)[0]

    levels = {0:"Beginner",1:"Intermediate",2:"Advanced"}
    st.success(f"Predicted Skill Level: {levels[pred]}")

    skill_name = levels[pred]

    # Select rackets for skill level
    rec_rackets = rackets[rackets["Skill"] == skill_name].sample(1)
    rec_shoes = shoes[shoes["Skill"] == skill_name].sample(1)

    st.subheader("🎯 Recommended Equipment")

    st.write(f"🏸 **Racket:** {rec_rackets.iloc[0]['Brand']} {rec_rackets.iloc[0]['Model']} ({rec_rackets.iloc[0]['Type']})")
    st.write(f"👟 **Shoes:** {rec_shoes.iloc[0]['Brand']} {rec_shoes.iloc[0]['Model']} (Cushion: {rec_shoes.iloc[0]['Cushion']})")
    # Radar Chart
    st.subheader("📊 Skill Profile")

    labels = np.array(["Fitness","Footwork","Smash","Reaction"])
    values = np.array([fitness, footwork, smash, reaction])

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values = np.concatenate((values,[values[0]]))
    angles = np.concatenate((angles,[angles[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    ax.set_title("Player Skill Radar")
    ax.set_ylim(0,10)

    st.pyplot(fig)
    # -----------------------
    # Growth Prediction
    # -----------------------
    st.subheader("⏳ Performance Growth Estimate")

    months = growth_model.predict(new_player)[0]
    st.info(f"Estimated time to reach next skill level: {months:.1f} months")
    # -----------------------
    # TRAINING FOCUS AI
    # -----------------------
    st.subheader("🎯 Training Focus Suggestion")

    skills = {
    "Fitness": fitness,
    "Footwork": footwork,
    "Smash": smash,
    "Reaction": reaction
    }

    weakest_skill = min(skills, key=skills.get)

    focus_tips = {
    "Fitness": "Focus on stamina drills, skipping, and endurance training.",
    "Footwork": "Practice shadow badminton and multi-shuttle movement drills.",
    "Smash": "Work on arm strength and smash technique drills.",
    "Reaction": "Do reflex drills and fast defense training."
    }

    st.warning(f"Priority Training Area: **{weakest_skill}**")
    st.write(focus_tips[weakest_skill])

    # -----------------------
    # SAVE PLAYER TO DATABASE
    # -----------------------
    new_record = pd.DataFrame({
    "Fitness":[fitness],
    "Footwork":[footwork],
    "Smash":[smash],
    "Reaction":[reaction],
    "Skill_Level":[skill_name],
    "Racket":[rec_rackets.iloc[0]['Model']],
    "Shoes":[rec_shoes.iloc[0]['Model']],
    "Months_To_Next_Level":[round(months,1)]
    })

    new_record.to_csv(db_file, mode='a', header=False, index=False)

    st.success("Player data saved to database!")
# -----------------------
# COACH DASHBOARD
# -----------------------
st.header("📊 Coach Dashboard")

if os.path.exists(db_file):
    player_data = pd.read_csv(db_file)
    st.dataframe(player_data)
else:
    st.write("No player data available yet.")



