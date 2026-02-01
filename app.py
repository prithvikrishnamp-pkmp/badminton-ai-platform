import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import os
import sys
import os
sys.path.append(os.path.dirname(__file__))
import pose_analysis

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on unseen data
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.write(f"Test Accuracy: {acc*100:.2f}%")
st.write("Confusion Matrix:")
st.write(cm)

# ==============================
# INJURY RISK TRAINING DATA
# ==============================

injury_data = {
    "Training Load": [8,7,9,6,5,10,4,8,7,9],
    "Fatigue": [7,6,8,5,4,9,3,7,6,8],
    "Recovery": [3,4,2,5,6,1,7,3,4,2],
    "Risk": [2,1,2,1,0,2,0,1,1,2]  # 0 Low, 1 Medium, 2 High
}

inj_df = pd.DataFrame(injury_data)

X_injury = inj_df[["Training Load", "Fatigue", "Recovery"]]
y_injury = inj_df["Risk"]

injury_model = RandomForestClassifier()
injury_model.fit(X_injury, y_injury)

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
# ==============================
# INJURY RISK TRAINING DATA
# ==============================

injury_data = {
    "Training Load": [8,7,9,6,5,10,4,8,7,9],
    "Fatigue": [7,6,8,5,4,9,3,7,6,8],
    "Recovery": [3,4,2,5,6,1,7,3,4,2],
    "Risk": [2,1,2,1,0,2,0,1,1,2]
}

inj_df = pd.DataFrame(injury_data)

X_injury = inj_df[["Training Load", "Fatigue", "Recovery"]]
y_injury = inj_df["Risk"]

injury_model = RandomForestClassifier()
injury_model.fit(X_injury, y_injury)


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
    st.subheader("🏋️ Training Recommendations")

    if footwork < 5:
        st.write("• Improve lateral movement drills and shadow practice")
    if smash < 5:
        st.write("• Focus on smash power and wrist strengthening")
    if reaction < 5:
        st.write("• Add reflex training and multi-shuttle drills")
    if fitness < 5:
        st.write("• Improve endurance and on-court stamina training")


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
    # Injury Risk Prediction
    st.subheader("⚠️ Injury Risk Assessment")

    # Create proper input for injury model
    injury_input = pd.DataFrame({
    "Training Load": [fitness + smash],
    "Fatigue": [10 - fitness],
    "Recovery": [reaction]
    })

    risk_pred = injury_model.predict(injury_input)[0]
    st.subheader("⚠️ Fatigue Analysis")

    if fitness < 4 and smash > 7:
        st.warning("High workload with low fitness — Risk of overtraining")
    elif fitness > 7 and reaction > 7:
        st.success("Good recovery and conditioning level")
    else:
        st.info("Balanced training load")




    risk_levels = {0:"Low Risk",1:"Medium Risk",2:"High Risk"}

    st.warning(f"Injury Risk Level: **{risk_levels[risk_pred]}**")

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
    # AI TRAINING PLAN
    # -----------------------
    st.subheader("📅 Personalized Weekly Training Plan")

    plan = {
    "Fitness":"Cardio + endurance drills",
    "Footwork":"Shadow badminton + multi-shuttle",
    "Smash":"Strength + smash technique",
    "Reaction":"Reflex and defense drills"
    }

    st.write(f"**Monday:** {plan[weakest_skill]}")
    st.write("**Tuesday:** Match simulation")
    st.write("**Wednesday:** Skill correction drills")
    st.write("**Thursday:** Footwork + speed work")
    st.write("**Friday:** Smash & net play")
    st.write("**Saturday:** Practice match")
    st.write("**Sunday:** Recovery & stretching")


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

    # Progress Graph
    st.subheader("📈 Player Progress Trend")

    if len(player_data) > 1:
        player_data["Skill_Num"] = player_data["Skill_Level"].map({
            "Beginner":1,
            "Intermediate":2,
            "Advanced":3
        })

        st.line_chart(player_data["Skill_Num"])
    else:
        st.write("Not enough data for progress tracking yet.")

else:
    st.write("No player data available yet.")


# -----------------------------
# 🎥 AI VIDEO POSE + MOVEMENT ANALYSIS
# -----------------------------
st.header("🎥 Upload Badminton Video for AI Analysis")

uploaded_video = st.file_uploader("Upload badminton practice video", type=["mp4", "mov"])

if uploaded_video is not None:
    st.video(uploaded_video)

    import tempfile
    


    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    st.write("Processing video with AI... ⏳")

    pose_analysis.analyze_video(tfile.name)

    st.success("Analysis Complete ✅")
