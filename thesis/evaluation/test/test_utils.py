from thesis.evaluation.utils import pick_steps_q_half, pick_steps_h_grid

# pick_steps_q_half("../evaluation-results/QSC-EGO-1", include_inbetween_steps=True)

pick_steps_h_grid("../../evaluation-results/QSC-EGO-1", list(range(0, 11)) + [20, 30, 40, 50])