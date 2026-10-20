/// Per-series smoothed mean-target encoding for categorical covariates.
pub fn target_encode(
    target_context: &[f64],
    past_codes: &[f64],
    future_codes: Option<&[f64]>,
    smooth: f64,
) -> (Vec<f32>, Option<Vec<f32>>) {
    assert_eq!(target_context.len(), past_codes.len());
    let mut observed_count = 0.0f64;
    let mut target_sum = 0.0f64;
    for &value in target_context {
        if value.is_finite() {
            observed_count += 1.0;
            target_sum += value;
        }
    }
    let item_mean = if observed_count > 0.0 {
        target_sum / observed_count
    } else {
        0.0
    };

    let mut categories: Vec<f64> = past_codes
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    categories.sort_by(|a, b| a.partial_cmp(b).unwrap());
    categories.dedup_by(|a, b| a == b);
    let nan_slot = categories.len();
    let unseen = nan_slot + 1;

    let lookup_index = |code: f64| -> usize {
        if code.is_nan() {
            return nan_slot;
        }
        match categories.binary_search_by(|probe| probe.partial_cmp(&code).unwrap()) {
            Ok(index) => index,
            Err(_) => unseen,
        }
    };

    let mut sums = vec![0.0f64; nan_slot + 1];
    let mut counts = vec![0.0f64; nan_slot + 1];
    let mut past_slots = Vec::with_capacity(past_codes.len());
    for (index, &code) in past_codes.iter().enumerate() {
        let slot = lookup_index(code);
        debug_assert!(slot != unseen, "every past code must be in the categories");
        past_slots.push(slot);
        let observed = target_context[index].is_finite();
        if observed {
            sums[slot] += target_context[index];
            counts[slot] += 1.0;
        }
    }
    let mut lookup = Vec::with_capacity(nan_slot + 2);
    for slot in 0..=nan_slot {
        lookup.push((smooth * item_mean + sums[slot]) / (smooth + counts[slot]));
    }
    lookup.push(item_mean);

    let encoded_past = past_slots
        .iter()
        .map(|&slot| lookup[slot] as f32)
        .collect::<Vec<_>>();
    let encoded_future = future_codes.map(|codes| {
        codes
            .iter()
            .map(|&code| lookup[lookup_index(code)] as f32)
            .collect::<Vec<_>>()
    });
    (encoded_past, encoded_future)
}
