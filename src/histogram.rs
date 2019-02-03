trait Histogram {
    // add code here

    static STATIC: Type = init;
}

trait HistogramBuckets {
    fn num_buckets(&self) -> u16;

    /// Gets the bucket definition for number no.  Observations for values <= this bucket, so it represents
    /// an upper bound.
    fn bucket_top(&self, no: u16) -> f64;
}

struct GeometricBuckets {
    numbuckets: u16,
    initial_bucket: f64,
    multiplier: f64,
}