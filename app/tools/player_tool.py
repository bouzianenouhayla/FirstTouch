import pandas as pd
from langsmith import traceable
from statsbombpy import sb

COMPETITION_NAME_TO_ID: dict[str, int] = {
    "FIFA Women's World Cup": 72,
    "FA Women's Super League": 37,
    "NWSL": 49,
    "UEFA Women's Euro": 53,
}


def _latest_season(comps: pd.DataFrame, competition_id: int) -> tuple[int, str] | None:
    """Return (season_id, season_name) for the most recent season of a competition.

    Args:
        comps: Full competitions DataFrame from statsbombpy.
        competition_id: StatsBomb competition ID.

    Returns:
        Tuple of (season_id, season_name), or None if not found.
    """
    subset = comps[comps["competition_id"] == competition_id].sort_values(
        "season_name", ascending=False
    )
    if subset.empty:
        return None
    row = subset.iloc[0]
    return int(row["season_id"]), str(row["season_name"])


def _player_appeared(positions: object) -> bool:
    """Check if a player actually played based on their positions entry.

    Args:
        positions: Positions value from lineups DataFrame (list or empty list).

    Returns:
        True if the player appeared on the pitch.
    """
    if isinstance(positions, list):
        return len(positions) > 0
    return False


@traceable(name="tool:search_player")
def search_player(player_name: str, competition: str | None = None) -> str:
    """Search women's football player appearances from StatsBomb open data.

    Args:
        player_name: Player name to search for (case-insensitive partial match).
        competition: Optional competition name to narrow the search.

    Returns:
        Formatted string with player appearances, or a not-found message.
    """
    try:
        all_comps = sb.competitions()
        women = all_comps[all_comps["competition_gender"] == "female"]

        target_ids = (
            [COMPETITION_NAME_TO_ID[competition]]
            if competition and competition in COMPETITION_NAME_TO_ID
            else list(COMPETITION_NAME_TO_ID.values())
        )

        appearances: list[str] = []
        search_term = player_name.lower()

        for cid in target_ids:
            result = _latest_season(women, cid)
            if result is None:
                continue
            sid, sname = result
            cname_rows = women[women["competition_id"] == cid]
            cname = (
                str(cname_rows.iloc[0]["competition_name"])
                if not cname_rows.empty
                else str(cid)
            )

            try:
                matches = sb.matches(competition_id=cid, season_id=sid)
            except Exception:
                continue

            for _, match in matches.iterrows():
                match_id = int(match["match_id"])
                date = match.get("match_date", "?")
                home = str(match.get("home_team", "?"))
                away = str(match.get("away_team", "?"))

                stage = str(match.get("competition_stage", ""))
                if isinstance(match.get("competition_stage"), dict):
                    stage = match["competition_stage"].get("name", "")

                try:
                    lineups = sb.lineups(match_id=match_id)
                except Exception:
                    continue

                for team_name, lineup_df in lineups.items():
                    matched = lineup_df[
                        lineup_df["player_name"]
                        .str.lower()
                        .str.contains(search_term, na=False)
                    ]
                    for _, player_row in matched.iterrows():
                        played = _player_appeared(player_row.get("positions"))
                        status = "played" if played else "squad (unused sub)"
                        stage_label = f" [{stage}]" if stage else ""
                        appearances.append(
                            f"{cname} {sname}{stage_label} | {date} | {home} vs {away} "
                            f"| {player_row['player_name']} ({team_name}) — {status}"
                        )

        if not appearances:
            return f"No appearances found for '{player_name}'."
        return f"Appearances for '{player_name}':\n" + "\n".join(appearances)

    except Exception as exc:
        return f"Error fetching player data: {exc}"
