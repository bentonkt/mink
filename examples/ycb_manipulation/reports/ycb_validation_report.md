# YCB Manipulation Benchmark

Generated: 2026-02-20T00:09:24.198069+00:00
YCB base: `/Users/bentontameling/Dev/ycb-tools/models/ycb`
Objects: 95
Pass: 63
Fail: 32
Library JSON: `examples/ycb_manipulation/reports/ycb_manipulation_library.json`

## Failed Objects

- `001_chips_can`: release
- `011_banana`: settle
- `019_pitcher_base`: grasp, release
- `022_windex_bottle`: settle
- `023_wine_glass`: settle, grasp
- `026_sponge`: settle
- `029_plate`: release
- `031_spoon`: contact
- `033_spatula`: grasp, release
- `035_power_drill`: release
- `040_large_marker`: grasp, release
- `042_adjustable_wrench`: settle
- `049_small_clamp`: settle, contact
- `052_extra_large_clamp`: grasp
- `055_baseball`: settle
- `057_racquetball`: grasp
- `059_chain`: release
- `063-b_marbles`: grasp, release
- `065-a_cups`: settle
- `065-b_cups`: settle
- `065-c_cups`: settle, grasp
- `065-d_cups`: settle
- `070-a_colored_wood_blocks`: settle
- `070-b_colored_wood_blocks`: settle
- `071_nine_hole_peg_test`: release
- `072-a_toy_airplane`: settle
- `072-h_toy_airplane`: settle, grasp, contact
- `073-e_lego_duplo`: grasp
- `073-f_lego_duplo`: grasp
- `073-g_lego_duplo`: settle
- `073-h_lego_duplo`: settle
- `073-m_lego_duplo`: release

## Per-Object Summary

| Object | Pass | Grasp | Release | Settle | Contact | Min Contact Dist |
|---|---:|---:|---:|---:|---:|---:|
| `001_chips_can` | N | Y | N | Y | Y | -0.00888 |
| `002_master_chef_can` | Y | Y | Y | Y | Y | -0.01330 |
| `003_cracker_box` | Y | Y | Y | Y | Y | -0.01288 |
| `004_sugar_box` | Y | Y | Y | Y | Y | -0.01849 |
| `005_tomato_soup_can` | Y | Y | Y | Y | Y | -0.01195 |
| `006_mustard_bottle` | Y | Y | Y | Y | Y | -0.01357 |
| `007_tuna_fish_can` | Y | Y | Y | Y | Y | -0.01831 |
| `008_pudding_box` | Y | Y | Y | Y | Y | -0.00834 |
| `009_gelatin_box` | Y | Y | Y | Y | Y | -0.01180 |
| `010_potted_meat_can` | Y | Y | Y | Y | Y | -0.01591 |
| `011_banana` | N | Y | Y | N | Y | -0.01661 |
| `012_strawberry` | Y | Y | Y | Y | Y | -0.00841 |
| `013_apple` | Y | Y | Y | Y | Y | -0.00720 |
| `014_lemon` | Y | Y | Y | Y | Y | -0.00824 |
| `015_peach` | Y | Y | Y | Y | Y | -0.00836 |
| `016_pear` | Y | Y | Y | Y | Y | -0.00957 |
| `017_orange` | Y | Y | Y | Y | Y | -0.00821 |
| `018_plum` | Y | Y | Y | Y | Y | -0.00900 |
| `019_pitcher_base` | N | N | N | Y | Y | -0.01037 |
| `021_bleach_cleanser` | Y | Y | Y | Y | Y | -0.00901 |
| `022_windex_bottle` | N | Y | Y | N | Y | -0.01474 |
| `023_wine_glass` | N | N | Y | N | Y | -0.02195 |
| `024_bowl` | Y | Y | Y | Y | Y | -0.02210 |
| `025_mug` | Y | Y | Y | Y | Y | -0.01184 |
| `026_sponge` | N | Y | Y | N | Y | -0.02158 |
| `028_skillet_lid` | Y | Y | Y | Y | Y | -0.02795 |
| `029_plate` | N | Y | N | Y | Y | -0.02648 |
| `030_fork` | Y | Y | Y | Y | Y | -0.02775 |
| `031_spoon` | N | Y | Y | Y | N | -0.03000 |
| `032_knife` | Y | Y | Y | Y | Y | -0.01095 |
| `033_spatula` | N | N | N | Y | Y | -0.02139 |
| `035_power_drill` | N | Y | N | Y | Y | -0.02257 |
| `036_wood_block` | Y | Y | Y | Y | Y | -0.01350 |
| `037_scissors` | Y | Y | Y | Y | Y | -0.01456 |
| `038_padlock` | Y | Y | Y | Y | Y | -0.01182 |
| `040_large_marker` | N | N | N | Y | Y | -0.02251 |
| `041_small_marker` | Y | Y | Y | Y | Y | -0.01242 |
| `042_adjustable_wrench` | N | Y | Y | N | Y | -0.01410 |
| `043_phillips_screwdriver` | Y | Y | Y | Y | Y | -0.00902 |
| `044_flat_screwdriver` | Y | Y | Y | Y | Y | -0.01271 |
| `048_hammer` | Y | Y | Y | Y | Y | -0.02286 |
| `049_small_clamp` | N | Y | Y | N | N | -0.03746 |
| `050_medium_clamp` | Y | Y | Y | Y | Y | -0.00745 |
| `051_large_clamp` | Y | Y | Y | Y | Y | -0.01243 |
| `052_extra_large_clamp` | N | N | Y | Y | Y | -0.02224 |
| `053_mini_soccer_ball` | Y | Y | Y | Y | Y | -0.00608 |
| `054_softball` | Y | Y | Y | Y | Y | -0.00752 |
| `055_baseball` | N | Y | Y | N | Y | -0.00618 |
| `056_tennis_ball` | Y | Y | Y | Y | Y | -0.00694 |
| `057_racquetball` | N | N | Y | Y | Y | -0.00647 |
| `058_golf_ball` | Y | Y | Y | Y | Y | -0.00731 |
| `059_chain` | N | Y | N | Y | Y | -0.02601 |
| `061_foam_brick` | Y | Y | Y | Y | Y | -0.01788 |
| `062_dice` | Y | Y | Y | Y | Y | -0.00844 |
| `063-a_marbles` | Y | Y | Y | Y | Y | -0.00839 |
| `063-b_marbles` | N | N | N | Y | Y | -0.01732 |
| `063-d_marbles` | Y | Y | Y | Y | Y | -0.00952 |
| `065-a_cups` | N | Y | Y | N | Y | -0.01681 |
| `065-b_cups` | N | Y | Y | N | Y | -0.01440 |
| `065-c_cups` | N | N | Y | N | Y | -0.01771 |
| `065-d_cups` | N | Y | Y | N | Y | -0.01091 |
| `065-e_cups` | Y | Y | Y | Y | Y | -0.02576 |
| `065-f_cups` | Y | Y | Y | Y | Y | -0.01175 |
| `065-g_cups` | Y | Y | Y | Y | Y | -0.01279 |
| `065-h_cups` | Y | Y | Y | Y | Y | -0.01280 |
| `065-i_cups` | Y | Y | Y | Y | Y | -0.00763 |
| `065-j_cups` | Y | Y | Y | Y | Y | -0.01487 |
| `070-a_colored_wood_blocks` | N | Y | Y | N | Y | -0.01004 |
| `070-b_colored_wood_blocks` | N | Y | Y | N | Y | -0.01262 |
| `071_nine_hole_peg_test` | N | Y | N | Y | Y | -0.01264 |
| `072-a_toy_airplane` | N | Y | Y | N | Y | -0.01909 |
| `072-b_toy_airplane` | Y | Y | Y | Y | Y | -0.02443 |
| `072-c_toy_airplane` | Y | Y | Y | Y | Y | -0.00865 |
| `072-d_toy_airplane` | Y | Y | Y | Y | Y | -0.00893 |
| `072-e_toy_airplane` | Y | Y | Y | Y | Y | -0.00898 |
| `072-f_toy_airplane` | Y | Y | Y | Y | Y | -0.00927 |
| `072-h_toy_airplane` | N | N | Y | N | N | -0.07747 |
| `072-i_toy_airplane` | Y | Y | Y | Y | Y | -0.00825 |
| `072-j_toy_airplane` | Y | Y | Y | Y | Y | -0.00727 |
| `072-k_toy_airplane` | Y | Y | Y | Y | Y | -0.00846 |
| `073-a_lego_duplo` | Y | Y | Y | Y | Y | -0.01531 |
| `073-b_lego_duplo` | Y | Y | Y | Y | Y | -0.00870 |
| `073-c_lego_duplo` | Y | Y | Y | Y | Y | -0.00642 |
| `073-d_lego_duplo` | Y | Y | Y | Y | Y | -0.00726 |
| `073-e_lego_duplo` | N | N | Y | Y | Y | -0.01796 |
| `073-f_lego_duplo` | N | N | Y | Y | Y | -0.01972 |
| `073-g_lego_duplo` | N | Y | Y | N | Y | -0.02399 |
| `073-h_lego_duplo` | N | Y | Y | N | Y | -0.00929 |
| `073-i_lego_duplo` | Y | Y | Y | Y | Y | -0.01531 |
| `073-j_lego_duplo` | Y | Y | Y | Y | Y | -0.01359 |
| `073-k_lego_duplo` | Y | Y | Y | Y | Y | -0.01656 |
| `073-l_lego_duplo` | Y | Y | Y | Y | Y | -0.01471 |
| `073-m_lego_duplo` | N | Y | N | Y | Y | -0.01954 |
| `076_timer` | Y | Y | Y | Y | Y | -0.00675 |
| `077_rubiks_cube` | Y | Y | Y | Y | Y | -0.00716 |
