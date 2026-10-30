import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    page_icon="📦",
    layout="wide"
)

# دالة تحميل النموذج المحفوظ
@st.cache_resource
def get_model():
    return joblib.load("wholesale_model.joblib")

try:
    model_data = get_model()
    pca = model_data["pca"]
    clf = model_data["classifier"]
    cluster_map = model_data["clusters_map"]
    ready = True
except Exception as e:
    ready = False
    st.error(f"⚠️ يرجى التأكد من تشغيل train_and_save.py أولاً لإنشاء ملف wholesale_model.joblib. الخطأ: {e}")

st.title("📦 لوحة تصنيف عملاء الجملة واتخاذ القرارات اللوجستية")
st.markdown("توقع تصنيف العميل وتحديد استراتيجية الشحن المثلى بناءً على حجم المشتريات السنوية.")

if ready:
    # الشريط الجانبي لإدخال البيانات
    st.sidebar.header("🎯 مدخلات إنفاق العميل السنوي ($)")
    
    preset = st.sidebar.selectbox(
        "نماذج معدّة مسبقاً:",
        ["مخصص", "سوبرماركت ضخم", "مطعم فاخر", "مقهى"]
    )
    
    presets_dict = {
        "مخصص": [10000, 2000, 3000, 1500, 500, 1000],
        "سوبرماركت ضخم": [3500, 8500, 15000, 900, 5200, 1200],
        "مطعم فاخر": [18000, 1800, 2200, 4500, 250, 2100],
        "مقهى": [3000, 4500, 5000, 600, 1200, 450]
    }
    vals = presets_dict[preset]

    fresh = st.sidebar.number_input("Fresh (منتجات طازجة)", min_value=1.0, value=float(vals[0]), step=500.0)
    milk = st.sidebar.number_input("Milk (ألبان)", min_value=1.0, value=float(vals[1]), step=500.0)
    grocery = st.sidebar.number_input("Grocery (بقالة معلبة)", min_value=1.0, value=float(vals[2]), step=500.0)
    frozen = st.sidebar.number_input("Frozen (مجمدات)", min_value=1.0, value=float(vals[3]), step=500.0)
    detergents = st.sidebar.number_input("Detergents_Paper (منظفات وورقيات)", min_value=1.0, value=float(vals[4]), step=500.0)
    delicatessen = st.sidebar.number_input("Delicatessen (لحوم ومقبلات)", min_value=1.0, value=float(vals[5]), step=500.0)

    # تجهيز المدخلات
    raw_df = pd.DataFrame([{
        'Fresh': fresh, 'Milk': milk, 'Grocery': grocery,
        'Frozen': frozen, 'Detergents_Paper': detergents, 'Delicatessen': delicatessen
    }])
    log_df = np.log(raw_df)

    # التنبؤ وحساب إحداثيات PCA
    pca_point = pca.transform(log_df)[0]
    pred_cluster = int(clf.predict(log_df)[0])
    probs = clf.predict_proba(log_df)[0]
    confidence = float(np.max(probs))

    # عرض النتائج الرئيسية
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("القطاع المتوقع", cluster_map[pred_cluster]["name"])
    with col2:
        st.metric("دقة التوقع", f"{confidence * 100:.1f}%")
    with col3:
        st.metric("جدول الشحن الموصى به", f"{cluster_map[pred_cluster]['delivery_days']} أيام / أسبوع")

    st.markdown("---")

    # الرسوم البيانية
    left_plot, right_plot = st.columns(2)
    
    with left_plot:
        st.subheader("📍 إسقاط العميل على محاور PCA")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(pca_point[0], pca_point[1], color='crimson', s=180, marker='X', label='العميل الحالي')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        ax.set_xlabel("Dimension 1 (Grocery / Paper)")
        ax.set_ylabel("Dimension 2 (Fresh / Frozen)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)

    with right_plot:
        st.subheader("📊 توزيع الإنفاق حسب الفئات")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=raw_df.columns, y=raw_df.values[0], palette="crest", ax=ax2)
        ax2.set_ylabel("القيمة ($)")
        ax2.set_xticklabels(raw_df.columns, rotation=30, ha='right')
        st.pyplot(fig2)

    # تفاصيل القرار التجاري
    st.subheader("💡 التوصيات اللوجستية وإدارة المخزون:")
    if pred_cluster == 0:
        st.info(f"**{cluster_map[0]['name']}**: تركيز عالي على البضائع غير القابلة للتلف السريع. يمكن خفض مواعيد الشحن إلى **{cluster_map[0]['delivery_days']} أيام أسبوعياً** لتقليل تكاليف النقل دون التأثير على عمليات العميل.")
    else:
        st.success(f"**{cluster_map[1]['name']}**: استهلاك كثيف للمنتجات الطازجة والمجمدة يومياً. يوصى بجدول شحن **{cluster_map[1]['delivery_days']} أيام أسبوعياً** لتفادي تلف البضائع أو نفاد المخزون التشغيلي.")