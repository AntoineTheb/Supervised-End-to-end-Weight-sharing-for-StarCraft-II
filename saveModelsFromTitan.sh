source secret_config.sh
if [ -z "$TITAN" ] || [ -z "$TITANDIR" ]
then
	echo "You must define TITAN and TITANDIR"
	echo "Add them to secret_config.sh (see on slack)"
	echo "DON'T TRACK secret_config.sh IN GIT!!!"
	echo ""
	echo "secret_config.sh"
	echo "   TITAN=username@address"
	echo "   TITANDIR=path_on_titan_for_your_stuff"
else
	rsync -rvtze ssh $TITAN:$TITANDIR/bin/ ./bin --progress
fi