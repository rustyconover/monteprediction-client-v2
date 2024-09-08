from monteprediction import SPDR_ETFS
from monteprediction.submission import send_in_chunks
import numpy as np

from sklearn.decomposition import PCA
import yfinance as yf
import pandas as pd
from scipy.stats.qmc import MultivariateNormalQMC
from sklearn.covariance import EmpiricalCovariance
from monteprediction.calendarutil import get_last_wednesday
from monteprediction.submission import send_in_chunks

from sklearn.covariance import LedoitWolf

from datetime import timedelta
import os

# Factory defaults (don't modify)
num_samples_per_chunk = int(1048576 / 8)
num_chunks = 8
num_samples = num_chunks * num_samples_per_chunk


def generate_sample_with_weeks(
    lookback_num_weeks: int, use_zero_mean: bool, estimator
) -> pd.DataFrame:
    last_wednesday = get_last_wednesday()
    lookback_num_weeks = 4
    start_date = last_wednesday - timedelta(weeks=lookback_num_weeks)
    data = yf.download(SPDR_ETFS, start=start_date, end=last_wednesday, interval="1wk")
    weekly_prices = data["Adj Close"]
    weekly_returns = weekly_prices.pct_change().dropna()

    assert sorted(SPDR_ETFS) == SPDR_ETFS

    # Use cov estimation to generate samples
    cov_matrix = estimator(weekly_returns)
    #    cov_matrix = EmpiricalCovariance().fit(weekly_returns).covariance_
    qmc_engine = MultivariateNormalQMC(
        mean=np.zeros(len(SPDR_ETFS)) if use_zero_mean else weekly_returns.mean(),
        cov=cov_matrix,
    )
    samples = qmc_engine.random(num_samples)
    df = pd.DataFrame(columns=SPDR_ETFS, data=samples)
    print(df[:3])

    # Verify submission
    assert len(df.index) == num_samples, f"Expecting exactly {num_samples} samples"
    assert list(df.columns) == SPDR_ETFS, "Columns should match SPDR_ETFS in order"

    return df


methods = {
    "empirical_covariance": lambda X: np.cov(X, rowvar=False),
    "sklearn_empirical_covariance": lambda X: EmpiricalCovariance().fit(X).covariance_,
    "pca_covariance": lambda X: PCA().fit(X).get_covariance(),
    "shrinkage_covariance": lambda X: LedoitWolf().fit(X).covariance_,
}


robot_names = [
    "gigglesnort",
    "wobbletron",
    "fizzlebots",
    "clankaroo",
    "jigglywump",
    "blipblorp",
    "zanytronix",
    "whizzbangbot",
    "fluffernutz",
    "snickerdroid",
    "twinklebot",
    "blorptastic",
    "wobblewump",
    "doodletron",
    "sprocketron",
    "gizmozilla",
    "fluffernutter",
    "gigglydroid",
    "bloopatron",
    "crankaroo",
    "fizzlebop",
    "zapfizz",
    "snorklewomp",
    "whirlydroid",
    "quirkbot",
    "bloopadoodle",
    "zippitybot",
    "wigglesnort",
    "grumbletron",
    "zizzlebot",
    "jollywumpus",
    "niftynutz",
    "puffernoodle",
    "splootabot",
    "blipzap",
    "gizmodoodle",
    "clunkertron",
    "twizzlepops",
    "gigglesnark",
    "doodleblorp",
    "bopplebot",
    "snickertron",
    "wobbleblip",
    "fluffymatic",
    "whizbangaroo",
    "dizzypop",
    "bloopinator",
    "zanyblorp",
    "jiggletron",
    "guffawbot",
    "clankersnort",
    "whirlyzap",
    "bloopnizzle",
    "snickerwump",
    "jigglywhizz",
    "gigglesnortz",
    "blorpadoodle",
    "sprocketzap",
    "wobblefizz",
    "fizzlerbot",
    "twistydroid",
    "bloopystix",
    "gigglewumpus",
    "clankazoid",
    "whizzblop",
    "zapplinator",
    "jigglyzap",
    "fluffernoodle",
    "sprocklebot",
    "wobblewhizz",
    "zanydoodle",
    "giggletronix",
    "snickerzap",
    "dizzynut",
    "blorpwhizz",
    "clanklebot",
    "twinklewump",
    "whirlyblorp",
    "gizmosnort",
    "gigglewhirly",
    "blipwomp",
    "sprocklewhizz",
    "zapfizzle",
    "whizzlematic",
    "clunkerwhizz",
    "gigglepop",
    "bloopotron",
    "jigglywobble",
    "snickerfizz",
    "fizzlenut",
    "blorpabot",
    "twistysnort",
    "clankletron",
    "whirlydoodle",
    "gizmotron",
    "gigglesplat",
    "bloopblip",
    "jigglyzany",
    "zapwomp",
    "fluffysnort",
    "snickerblop",
    "wobblepop",
    "fizzlematic",
    "doodlewhirly",
    "blipbot",
    "zanywhizz",
    "clunkerpop",
    "whizzdoodle",
    "giggleblorp",
    "zapfizzler",
]


assert os.environ["EMAIL"] is not None

for zero_mean in [True, False]:
    for cov_method in sorted(methods.keys()):
        for weeks in [2, 3, 4, 8, 32, 52, 52 * 2, 52 * 3]:
            name = robot_names.pop()
            label = f"weeks={weeks} estimator={cov_method} zero_mean={zero_mean}"
            print(f"Generating {label}")
            df = generate_sample_with_weeks(weeks, zero_mean, methods[cov_method])
            print(f"Sending {cov_method} - {label} to {os.environ['EMAIL']}")
            send_in_chunks(
                df, num_chunks=num_chunks, email=os.environ["EMAIL"], name=name
            )
            print("Sent")
