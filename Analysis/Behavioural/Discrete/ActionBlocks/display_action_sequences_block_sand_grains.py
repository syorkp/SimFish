from Analysis.Behavioural.Discrete.ActionBlocks.display_action_sequences_block import display_all_sequences
from Analysis.Behavioural.Tools.BehavLabels.extract_sand_grain_interaction_sequences import \
    get_sand_grain_engagement_sequences_multiple_trials

if __name__ == "__main__":
    #                            SAND GRAIN INTERACTION SEQUENCES
    seq = get_sand_grain_engagement_sequences_multiple_trials("dqn_scaffold_33-2", "Behavioural-Data-Free", "Naturalistic",
                                                              20, range_for_engagement=30, preceding_steps=20, proceeding_steps=10)
    seq = [s[:50] for s in seq]
    display_all_sequences(seq, min_length=50, max_length=60, save_figure=True, indicate_event_point=20,
                          figure_name="Sand-Grain-Interaction-dqn_scaffold_33-2",)
    #
    # #                            SAND GRAIN INTERACTION SEQUENCES (END)
    # seq = get_sand_grain_engagement_sequences_multiple_trials("dqn_scaffold_33-1", "Behavioural-Data-Free", "Naturalistic",
    #                                                           30, range_for_engagement=30, preceding_steps=20, proceeding_steps=20)
    # seq = [s[-50:] for s in seq]
    # display_all_sequences(seq, min_length=20, max_length=60, save_figure=True, indicate_event_point=20,
    #                       figure_save_location="Sand-Grain-Interaction-END-dqn_scaffold_33-1",)

    #                            PREY INTERACTION SEQUENCES
    # seq = get_paramecia_engagement_sequences_multiple_trials("dqn_scaffold_33-1", "Behavioural-Data-Free", "Naturalistic",
    #                                                           30, range_for_engagement=20, preceding_steps=20, proceeding_steps=10)
    # seq = [s for s in seq if len(s) < 50]
    #
    # display_all_sequences(seq, min_length=30, max_length=60, save_figure=True, indicate_event_point=20,
    #                       figure_save_location="Paramecia-Interaction-dqn_scaffold_33-1",)