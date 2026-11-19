# Downloads one NSRR folder with the nsrr gem into the current folder. The token comes from
# $NSRR_TOKEN (a SkyPilot secret), so it is never on a command line where `ps` on a shared node
# could show it, and never in a log, a task YAML or a repo.
#
#   ruby nsrr_fetch.rb DATASET/PATH [FILE_REGEX]
#   ruby nsrr_fetch.rb shhs/polysomnography/edfs/shhs1
#   ruby nsrr_fetch.rb mesa/polysomnography/edfs 'mesa-sleep-(0001|0002)\.edf'
require "nsrr"
require "nsrr/commands/download"

token = ENV.fetch("NSRR_TOKEN", "").strip
abort "nsrr_fetch: NSRR_TOKEN is not set" if token.empty?
path, regex = ARGV
abort "usage: ruby nsrr_fetch.rb DATASET/PATH [FILE_REGEX]" if path.to_s.empty?

args = ["download", path, "--token=#{token}", "--fast"] # --fast: skip files whose size already matches
args << "--file=#{regex}" if regex
Nsrr::Commands::Download.run(args)
